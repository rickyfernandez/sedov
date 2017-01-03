import matplotlib.pyplot as plt
import numpy as np
import sys

def InitialConditions(gamma):

    # initial conditions behind the shock
    f = (gamma + 1)/(gamma - 1)
    g = 4.0/(5.0*(gamma + 1))
    h = 8.0/(25.0*(gamma + 1))

    return [f, g, h]

def fPrime(x, f, g, h, gamma):

    # derivative of the dimensionless density
    fp = -5.0*f*(f*g*(5.*g - 2.*x)\
            *(10.*g - x) - 30.*h*x)
    fp /= x*(5.*g - 2*x)*(f*(5.*g-2.*x)**2\
            - 25.*gamma*h)

    return fp

def gPrime(x, f, g, h, gamma):

    # derivative of the dimensionless velocity
    gp  = 3.*f*g*(5.*g - 2.*x)*x\
            + h*(-30.*x + 50.*g*gamma)
    gp /= f*(5.*g - 2*x)**2*x -25.*h*x*gamma

    return gp

def hPrime(x, f, g, h, gamma):

    # derivative of the dimensionless pressure
    hp  = f*h*(5.*g*x*(6. + gamma)\
            - 12.*x**2 - 50.*g*g*gamma)
    hp /= f*(5.*g - 2*x)**2*x -25.*h*x*gamma

    return hp

def F(r, x, gamma):

    # present values of dimsionless density, 
    # velocity, and pressure
    f = r[0]
    g = r[1]
    h = r[2]

    # evaluate all derivates in vector form
    fp = fPrime(x, f, g, h, gamma)
    gp = gPrime(x, f, g, h, gamma)
    hp = hPrime(x, f, g, h, gamma)

    return np.array([fp, gp, hp])


def SolveBlastWave(N, gamma):

    # remember problem at the origin so stop
    # the integration close to the origin
    a  = 0.01     # inner boundary
    b  = 1.0      # outter boundary
    dx = (b-a)/N  # size of step

    # variables to hold density, velocity, and pressure solution
    f = []; g = []; h = []

    # generate radius points from the innner boundary
    # to the shock. Remember we are integrating from
    # the shock (outter boundary) to the inner boundary
    # so we have to reverse the points
    radius_range = np.arange(a, b+dx, dx)[::-1]

    # likewise our step size has to be negative such
    # that x+dx < x
    dx *= -1.0

    # intial values behind the shock to start the integration
    r = np.array(InitialConditions(gamma))

    # begin solution - advance from the shock to the inner
    # boundary
    for x in radius_range:

        # store time step solution
        f.append(r[0])
        g.append(r[1])
        h.append(r[2])

        # second order runge kutta method
        k1 = dx*F(r, x, gamma)
        k2 = dx*F(r + 0.5*k1, x + 0.5*dx, gamma)
        k3 = dx*F(r + 0.5*k2, x + 0.5*dx, gamma)
        k4 = dx*F(r + k3, x + dx, gamma)

        # update the solution
        r += (k1 + 2*k2 + 2*k3 + k4)/6.0

    # convert our result to numpy arrays and reverse the data so our
    # solution are ordered from the inner to outter boundary
    f = np.array(f)[::-1]; g = np.array(g)[::-1]; h = np.array(h)[::-1]
    radius_range = radius_range[::-1]

    # find the scaling constant by simpson integration
    integrand = 4*np.pi*(0.5*f*g**2 + h/(gamma-1.0))*radius_range**2

    xi = np.abs(dx)*(integrand[0] + integrand[-1] + 4.*np.sum(integrand[1:-1:2]) +\
            2.0*np.sum(integrand[2:-2:2]))/3.0
    xi = xi**(-1.0/5.0)

    return f, g, h, radius_range, xi


if __name__ == "__main__":

    # grab problem number from the terminal prompt
    problem = int(sys.argv[1])

    if problem == 1:

        N  = 1000         # number of integration intervals
        gamma = 5.0/3.0   # ratio of specific heats

        f, g, h, radius, xi = SolveBlastWave(N, gamma)

        # plot normalized solution - to normalize the solution
        # divide by the last element (value at the shock which 
        # is the largest value)
        plt.plot(radius, f/f[-1])
        plt.plot(radius, g/g[-1])
        plt.plot(radius, h/h[-1])

        plt.ylim(-0.1, 1.0)
        plt.xlim(0., 1.)

        l = plt.legend([r"$f(x)$", r"$g(x)$", r"$h(x)$"], loc=2, prop={'size':15})
        l.draw_frame(False)

        plt.xlabel(r"$r/R_s$", fontsize=15)
        plt.title(r"Blast Wave: $\xi$ = %0.3f" % xi)
        plt.savefig("Blast_wave.png")
        plt.show()


    if problem == 2:

        N  = 1000         # number of integration steps
        gamma = 5.0/3.0   # ratio of specific heats

        f, g, h, radius_range, xi = SolveBlastWave(N, gamma)

        # now map values to a rectangular grid - first 
        # pick the size of the box which is centered at the origin
        box_size = 1.0

        # to make the caluclation easier lets take the energy
        # equal to one, E=1, and the density equal to one, rho=1
        # now calculate the time needed for the shock to reach 
        # the edge of the box

        # time for the shock to reach the edge of the box (0.5*box_size)
        tfinal = (0.5*box_size/xi)**(5.0/2.0)

        time_steps = 10                               # how many time steps
        time = np.linspace(0.0, tfinal, time_steps)   # set of times
        Rn = xi*time**(2.0/5.0)                      # radius at each time

        # add the value of the background density
        # remember we set the density to 1.0
        rho = np.append(f, np.array([1.0]))

        # create the grid to map our solution
        res = 256                                         # resolution of the image
        x = np.linspace(-0.5*box_size, 0.5*box_size, res) # equally space points in 1d
        X, Y = np.meshgrid(x,x);                          # 2d grid from set of 1d points
        D = np.sqrt(X**2 + Y**2).flatten()                # matrix of the distance form the
                                                          # origin to point X Y

        # create a 2d image for every time we picked
        n = 0
        for r in Rn:

            # radius points from the inner boundary to the
            # shock
            radius = r*radius_range         # create radius
            dr = np.mean(np.diff(radius))   # bin length

            # we have to handle values that fall outside
            # of our integration limits in our box
            r_bin = radius - 0.5*dr  # left side of the bins

            # add the last bin and one more bin to hanlde values outside
            # the shock
            r_bin = np.append(radius, np.array([radius[-1]+0.5*dr, 1.0]))
            r_bin[0] = 0.0  # handle points less then the inner boundary

            # our 2d density solution - flatten it for the calculation
            Density  = np.zeros(X.shape)
            Density  = Density.flatten()

            # bin each point distance with the our solution
            whichbin = np.digitize(D, r_bin)

            # now grab corresponding density
            for i,j in enumerate(whichbin):
                Density[i] = rho[j-1]

            # reshape density
            Density = Density.reshape(res, res)

            plt.pcolor(X, Y, Density)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-.5*box_size, .5*box_size)
            plt.ylim(-.5*box_size, .5*box_size)

            cb = plt.colorbar()
            cb.set_label('Density')
            plt.clim(0, 4.1)

            plt.savefig("BlastWave_" + `n`.zfill(4))
            plt.clf()
            n += 1
