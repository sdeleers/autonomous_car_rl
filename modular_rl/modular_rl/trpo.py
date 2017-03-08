from modular_rl import *
from theano.compile.nanguardmode import NanGuardMode


# ================================================================
# Trust Region Policy Optimization
# ================================================================

class TrpoUpdater(EzFlat, EzPickle):
    options = [
        ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
        ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ]

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)

        self.stochpol = stochpol
        self.cfg = cfg

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        EzFlat.__init__(self, params)

        # ob_no = stochpol.input
        inp1 = stochpol.input[0]
        inp2 = stochpol.input[1]
        act_na = probtype.sampled_variable()
        adv_n = T.vector("adv_n")

        # Probability distribution:
        prob_np = stochpol.get_output()
        oldprob_np = probtype.prob_variable()

        logp_n = probtype.loglikelihood(act_na, prob_np)
        oldlogp_n = probtype.loglikelihood(act_na, oldprob_np)
        # N = ob_no.shape[0]
        N = inp1.shape[0] + inp2.shape[0]

        # Policy gradient:
        surr = (-1.0 / N) * T.exp(logp_n - oldlogp_n).dot(adv_n)
        pg = flatgrad(surr, params)

        # # RMS Prop
        # self.rmsprop_cache = None

        prob_np_fixed = theano.gradient.disconnected_grad(prob_np)
        kl_firstfixed = probtype.kl(prob_np_fixed, prob_np).sum() / N
        grads = T.grad(kl_firstfixed, params)

        flat_tangent = T.fvector(name="flat_tan")
        shapes = [var.get_value(borrow=True).shape for var in params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(T.reshape(flat_tangent[start:start + size], shape))
            start += size
        gvp = T.add(*[T.sum(g * tangent) for (g, tangent) in zipsame(grads, tangents)])  # pylint: disable=E1111
        # Fisher-vector product
        fvp = flatgrad(gvp, params)

        ent = probtype.entropy(prob_np).mean()
        kl = probtype.kl(oldprob_np, prob_np).mean()

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        # args = [ob_no, act_na, adv_n, oldprob_np]
        args = [inp1, inp2, act_na, adv_n, oldprob_np]

        self.compute_policy_gradient = theano.function(args, pg, **FNOPTS)
        self.compute_losses = theano.function(args, losses, **FNOPTS)
        self.compute_fisher_vector_product = theano.function([flat_tangent] + args, fvp, **FNOPTS)

    def __call__(self, paths):
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        # prob_np is oldprob_np if you look in function above (see args = [ob_no, act_na, adv_n, oldprob_np])
        # args = (ob_no, action_na, advantage_n, prob_np)
        no_sensors = 0
        args = (ob_no[:, 0:ob_no.shape[1]-1-no_sensors], ob_no[:, ob_no.shape[1]-1-no_sensors:ob_no.shape[1]], action_na, advantage_n, prob_np)
        thprev = self.get_params_flat()

        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, *args) + cfg["cg_damping"] * p  # pylint: disable=E1101,W0640

        start_time_pg = time.time()
        g = self.compute_policy_gradient(*args)
        print("--- compute_policy_gradient:  %s seconds ---" % (time.time() - start_time_pg))
        losses_before = self.compute_losses(*args)
        if np.allclose(g, 0):
            print "got zero gradient. not updating"
        else:
            # RMS Prop update
            # if (self.rmsprop_cache == None):
            #     self.rmsprop_cache = np.zeros_like(thprev)
            # decay_rate = 0.99
            # learning_rate = 1e-4
            # self.rmsprop_cache = decay_rate * self.rmsprop_cache + (1 - decay_rate)*g**2
            # theta = thprev - learning_rate*g/(np.sqrt(self.rmsprop_cache) + 1e-5)
            # self.set_params_flat(theta)

            start_time_stepdir = time.time()
            stepdir = cg(fisher_vector_product, -g)
            print("--- stepdir:  %s seconds ---" % (time.time() - start_time_stepdir))
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / cfg["max_kl"])
            print "lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.set_params_flat(th)
                return self.compute_losses(*args)[0]  # pylint: disable=W0640

            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
            print "success", success
            self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname + "_before"] = lbefore
            out[lname + "_after"] = lafter
        return out


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print "fval before", fval
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print "a/e/r", actual_improve, expected_improve, ratio
        if ratio > accept_ratio and actual_improve > 0:
            print "fval after", newfval
            return True, xnew
    return False, x


def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print titlestr % ("iter", "residual norm", "soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print fmtstr % (i + 1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
    return x
