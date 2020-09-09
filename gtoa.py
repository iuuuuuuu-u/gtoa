import numpy as np
import heapq
import matplotlib.pyplot as plt


def function(x):
    y1 = 0
    for i in range(len(x)):
        y1 = y1 + (x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) + 10 )
    y = abs(0-y1)
    return y

m = 30
h = 15
imax = 100
dimen = 20
F = 2
rangelow = -5.12
rangehigh = 5.12


pop = np.zeros((m,dimen))
pop_fitness = np.zeros(m)

betterpop = np.zeros((h,dimen))
badpop = np.zeros((h,dimen))
better_fitness = np.zeros(h)
bad_fitness = np.zeros(h)

his_bestfit=np.zeros(imax)

for j in range(m):
    pop[j] = np.random.uniform(low=rangelow, high=rangehigh,size=(1, dimen))
    pop_fitness[j] = function(pop[j])
allbestpop,allbestfit = pop[pop_fitness.argmin()].copy(),pop_fitness.min()

for i in range(imax):
    print("The iteration is:", i + 1)

    popave = np.sum(pop)
    # for j in range(m):
    #     popave = popave + pop[j]
    popave = popave / 30



    pop_fitness1 = pop_fitness.flatten()
    pop_fitness1 = pop_fitness1.tolist()
    three = list(map(pop_fitness1.index, heapq.nsmallest(3, pop_fitness1)))
    X1 = pop[three[0]]
    X2 = pop[three[1]]
    X3 = pop[three[2]]
    Xave = (X1 + X2 + X3)/3
    fitX1 = function(X1)
    fitXave = function(Xave)
    if fitX1 > fitXave:
        Xtea = X1
    else:
        Xtea = Xave

    halfbetter = list(map(pop_fitness1.index, heapq.nsmallest(h, pop_fitness1)))
    halfbad = list(map(pop_fitness1.index, heapq.nlargest(h, pop_fitness1)))
    for l in range(h):
        betterpop[l] = pop[halfbetter[l]]
        badpop[l] = pop[halfbad[l]]
        better_fitness[l] = pop_fitness[halfbetter[l]]
        bad_fitness[l] = pop_fitness[halfbad[l]]
    betterpopold = betterpop.copy()
    badpopold = badpop.copy()

    for l in range(h):
        a = np.random.rand()
        b = np.random.rand()
        c = 1 - b
        d = np.random.rand()

        pop_newbetter1 = betterpop[l] + a * (Xtea - F * (b * popave + c * betterpop[l]))

        fitness_newbetter1 = function(pop_newbetter1)

        if fitness_newbetter1 < better_fitness[l]:
            better_fitness[l] = fitness_newbetter1
            betterpop[l] = pop_newbetter1

        pop_newbad1 = badpop[l] + 2 * d * (Xtea - badpop[l])
        fitness_newbad1 = function(pop_newbad1)
        if fitness_newbad1 < bad_fitness[l]:
            bad_fitness[l] = fitness_newbad1
            badpop[l] = pop_newbad1

    rangeM = list(range(0, h))
    for l in range(h):
        e = np.random.rand()
        g = np.random.rand()
        j1 = np.random.choice(rangeM)
        j2 = np.random.choice(rangeM)

        if better_fitness[l] < better_fitness[j1]:
            pop_newbetter2 = betterpop[l] + e * (betterpop[l] - betterpop[j1]) + g * (betterpop[l] - betterpopold[l])
        else:
            pop_newbetter2 = betterpop[l] - e * (betterpop[l] - betterpop[j1]) + g * (betterpop[l] - betterpopold[l])
        fitness_newbetter2 = function(pop_newbetter2)
        if fitness_newbetter2 < better_fitness[l]:
            better_fitness[l] = fitness_newbetter1
            betterpop[l] = pop_newbetter2

        if bad_fitness[l] < bad_fitness[j2]:
            pop_newbad2 = badpop[l] + e * (badpop[l] - badpop[j1]) + g * (badpop[l] - badpopold[l])
        else:
            pop_newbad2 = badpop[l] - e * (badpop[l] - badpop[j1]) + g * (badpop[l] - badpopold[l])
        fitness_newbad2 = function(pop_newbad2)
        if fitness_newbad2 < bad_fitness[l]:
            bad_fitness[l] = fitness_newbad2
            badpop[l] = pop_newbad2

    for l in range(h):
        pop[l] = betterpop[l]
        pop[h + l] = badpop[l]
        pop_fitness[l] = better_fitness[l]
        pop_fitness[h + l] = better_fitness[l]

    allbestpop, allbestfit = pop[pop_fitness.argmin()].copy(), pop_fitness.min()
    print("The best fitness is:", allbestfit)

    his_bestfit[i] = allbestfit


print("After iteration, the best pop is:", allbestpop)
print("After iteration, the best fitness is:", allbestfit)

mean = np.sum(pop_fitness)/m
std = np.std(pop_fitness)
print("After iteration, the mean fitness of the swarm is:","%e"%mean)
print("After iteration, the std fitness of the swarm is:","%e"%std)


fig=plt.figure(figsize=(12, 10), dpi=300)
plt.title('The change of best fitness',fontdict={'weight':'normal','size': 30})
x=range(1,101,1)
plt.plot(x,his_bestfit,color="red",label="GWO",linewidth=3.0, linestyle="-")
plt.tick_params(labelsize=25)
plt.xlim(0,101)
plt.ylim(0,300)
plt.xlabel("Epoch",fontdict={'weight':'normal','size': 30})
plt.ylabel("Fitness value",fontdict={'weight':'normal','size': 30})
plt.xticks(range(0,101,10))
plt.legend(loc="upper right",prop={'size':20})
plt.savefig("fit-GTOA.png")
plt.show()





