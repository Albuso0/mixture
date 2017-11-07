"""
module for the usual method of moments
"""
import sympy
import numpy as np
import scipy
from discrete_rv import DiscreteRV
from model_gm import ModelGM
import moments as mom

class MM():
    """
    class for the usual method of moments
    sigma = 1
    """
    def __init__(self, k, sigma=1):
        self.k = k
        self.sigma = sigma
        self.num_mm = 2*k-1
        self.transform = mom.hermite_transform_matrix(2*self.k-1, self.sigma)

    def estimate(self, samples):
        """
        estimate from the usual method of moments
        """
        samples = np.asarray(samples)
        m_raw = mom.empirical_moment(samples, 2*self.k-1)
        m_raw = np.mean(m_raw, axis=1).reshape((self.num_mm, 1))
        m_decon = np.dot(self.transform[0], m_raw)+self.transform[1]
        mm_rv = self.mom_symbol(m_decon.reshape(self.num_mm))
        return ModelGM(w=mm_rv.weights, x=mm_rv.atoms, std=self.sigma)

    def mom_symbol(self, moments):
        """ Symbolic solver for ordinary method of moments
        find k-point representative distribution

        Args:
        moments: a sequence of estimated moments, starting from degree 1
            only use first 2k-1 moments
        num_comps: number of atoms in the output RV

        Returns:
        U(finiteRV): estimated RV, whose first 2k-1 moments equal m

        """

        num = 2*self.k-1
        assert len(moments) >= num

        p_var = sympy.symbols('p0:%d'%(self.k), nonnegative=True, real=True)
        x_var = sympy.symbols('x0:%d'%(self.k), real=True)

        eqn = -1
        for i in range(self.k):
            eqn = eqn + p_var[i]
        equations = [eqn]

        for i in range(num):
            eqn = -moments[i]
            ########### truncate a real number to rational
            # eq = -nsimplify(m[i], tolerance=0.1)
            # eq = float('%.2g' % -m[i])
            for j in range(self.k):
                eqn = eqn + p_var[j]*(x_var[j]**(i+1))
            equations.append(eqn)

        var = [x for xs in zip(p_var, x_var) for x in xs] # format: [p1,x1,p2,x2,...]
        s_var = sympy.solve(equations, var)
        # s = solveset(equations,list(p)+list(x))
        # s = nonlinsolve(equations,list(p)+list(x))

        ########## some other tools in SymPy
        # print(equations)
        # s = solve_poly_system(equations,list(p)+list(x))
        # s = solve_triangulated(equations,list(p)+list(x))
        # s = solve_triangulated(equations,p[0],p[1],x[0],x[1])

        ########## test over simple rational coefficients
        # testEq = [x[0]*p[0] - 2*p[0], 2*p[0]**2 - x[0]**2]
        # s = solve_triangulated(testEq, x[0], p[0])

        if len(s_var) == 0:
            return DiscreteRV(w=[], x=[])
        else:
            return DiscreteRV(w=s_var[0][0::2], x=s_var[0][1::2])



    def mom_numeric(self, moments, start):
        """ numerical solver for ordinary method of moments
        find k-point representative distribution

        Args:
        moments: a sequence of estimated moments, starting from degree 1
            only use first 2k-1 moments
        start: initial guess. k = the number of components in start.

        Returns:
        U(finiteRV): estimated RV

        """
        k = len(start.p)
        assert k == self.k

        num = 2*k-1
        assert len(moments) >= num
        moments = moments[0:num]

        def equation(x_var):
            """
            local moments condition equation
            """
            model = DiscreteRV(x=x_var[0:k], w=np.append(x_var[k:], 1-np.sum(x_var[k:])))
            return model.moment(num)-moments

        x0_var = np.concatenate((start.x, start.p[0:-1]))
        x_var = scipy.optimize.fsolve(equation, x0_var)
        # x = scipy.optimize.broyden2(equation, x0)

        return DiscreteRV(x=x_var[0:k], w=np.append(x_var[k:], 1-np.sum(x_var[k:])))
