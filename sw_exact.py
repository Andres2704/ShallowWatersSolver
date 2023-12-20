import numpy as np
g = 9.81
class exact_solution():
    def __init__(self, sol_type) -> None:
        self.sol_type = sol_type

    def call_solution(self, x, tf, N):
        if self.sol_type == 'rarefaction':
            hl = 0.25
            hr = 0.5
            ul = 0.1
            xdiv = 0.5
            return self.rarefaction(x, hl, hr, ul, tf, xdiv)
        elif self.sol_type == 'shock':
            hl = 0.50
            hr = 0.25       
            ul = 0.1
            xdiv = 0.5
            return self.shock(x, hl, hr, ul, tf, xdiv) 
        elif self.sol_type == 'bump':
            hr = 0.33
            q0 = 0.18
            return self.bump(q0, hr, N) 
        else:
            return np.zeros(N+1)
        
    def rarefaction(self, x, hl, hr, ul, t, xdiv):
        ur = ul + 2*((g*hr)**0.5 - (g*hl)**0.5)
        Ug = np.array([hl, hl*ul])
        Ud = np.array([hr, hr*ur])
        U = np.array([0.0, 0.0])
        try:
            mu = (x-xdiv)/t
        except:
            mu = (x-xdiv)/1e-25

        if ((x - xdiv) <= (ul + (g*hl)**0.5)*t):
            U = Ug.copy()
        elif ((x - xdiv) >= (ur + (g*hr)**0.5)*t):
            U = Ud.copy()
        else:
            U[0] = (1/(9*g))*(mu - ul + 2*(g*hl)**0.5)**2
            U[1] = U[0]*(ul + 2*((g*U[0])**0.5 - (g*hl)**0.5))
        return U
    
    # def shock(self, x, hl, hr, ul, t, xdiv):
    #     a = -0.05
    #     ur=ul+(hr-hl)*np.sqrt(0.5*g*(1/hl+1/hr))
    #     Ug = np.array([hl, hl*ul])
    #     Ud = np.array([hr, hr*ur])
    #     U = np.array([0.0, 0.0])

    #     sigma=ul+hr*np.sqrt(0.5*g*(1/hl+1/hr))

    #     if ( (x-a)< sigma*t ):
    #         U=Ug.copy()
    #     else:
    #         U=Ud.copy()
    #     return U
    
    def shock(self, x, hl, hr, ul, t, xdiv):
        ur=ul+(hr-hl)*np.sqrt(0.5*g*(1/hl+1/hr))
        Ug = np.array([hl, hl*ul])
        Ud = np.array([hr, hr*ur])
        U = np.array([0.0, 0.0])

        sigma=ul+hr*np.sqrt(0.5*g*(1/hl+1/hr))

        if ( (x-xdiv)< sigma*t ):
            U=Ug.copy()
        else:
            U=Ud.copy()
        return U
    
    def bump(self, q0, h0, N):
        x = np.linspace(0, 20, N)
        B = np.zeros(N)
        W = np.zeros((2, N))
        for i in range(N):
            if (x[i]>8 and x[i]<12):
                B[i] = 0.2 - 0.05*(x[i] - 10)**2
        i=0
        Njump=0
        jump_in = [True]
        hc=B[0]
        i=N-1
        while i > 0:
            i=i-1
            if ((B[i-1]<B[i])and(B[i]>B[i+1])and(B[i]>hc)):
                hc=B[i]
                Njump=Njump+1
        jump=jump_in
        while len(jump)<Njump:
            jump=jump+jump_in
        jump[Njump-1]=True
        Njump=0

        hc = (q0**2/g)**(1/3)

        h = np.zeros(N)
        i = N-1
        h = h0 - B
        head = h[i] + B[i] + (0.5/g)*(q0**2/h[i]**2)
        Subcritical = True

        while(i > 0):
            i = i - 1
            if Subcritical:
                coeffs = [1, B[i]-head, 0, q0**2/(2*g)]
                h[i] = np.roots(coeffs)[0]
                if h[i]<hc:
                    Subcritical = False
            else:
                RH = True

                # Finging the index of maximum of B where the critical point is
                while (RH):
                    i = i-1
                    if(B[i-1]<B[i]):  
                        RH = not(jump[Njump])
                        Njump = Njump + 1
                        if (RH):
                            while(B[i-1]<B[i]):
                                i = i - 1 
                h[i] = hc
                head = h[i] + B[i] + (0.5/g)*(q0**2/h[i]**2)
                k = i

                # Solving the supercritical region where h<hc
                while(not(RH) and (k < N-2)):
                    k = k + 1
                    coeffs = [1,B[k]-(q0**2/(2*g*hc**2)) - hc - max(B),0,q0**2/(2*g)]
                    h[k] = np.roots(coeffs)[1]
                    if (h[k+1] > hc):
                        F1  = q0**2*(1/h[k]) + 0.5*g*h[k]**2
                        F2  = q0**2*(1/h[k+1]) + 0.5*g*h[k+1]**2
                        RH = F2>F1
                
                Subcritical=True

        W[0, :] = B + h
        W[1, :] = q0

        return W
