library("rgp")

# if we are to match our simple grammar, then we have the following:

# only these functions
funcs <- functionSet("+","*","/","sin","exp")

# a single variable
vars <- inputVariableSet("x")

# constants of values 1, 2, or 3
const = constantFactorySet(function() as.integer(floor(runif(1,1,4))))

# and a target of '1 / 3 + x + sin( x * x )', on the range [-10,10]
domain <- seq(from=-10, to=10, by=0.01)
target <- function(x) (1 / 3 + x + sin( x * x ))
fitness <- 
  function(f) {
    err <- sqrt(mean((f(domain) - target(domain))^2))
    if (is.na(err)) { Inf } else { err }
  };

# run genetic programming
res <- geneticProgramming(fitnessFunction = fitness,
                          inputVariables = vars,
                          functionSet = funcs,
                          constantSet = const,
                          stopCondition = makeTimeStopCondition(30))

bestFit <- res$population[[which.min(res$fitnessValues)]]

# we've probably either found it, or something very close
plot(domain, target(domain), type="l", col="black")
lines(domain, bestFit(domain), col="red")

# this time it has sin(3) + sin(3), instead of (1/3).
# that's pretty close (approx 0.28, instead of 0.33).
print(res$population[which.min(res$fitnessValues)])

# function (x) 
#  x + sin(3L) + (sin(x * x) + sin(3L))
