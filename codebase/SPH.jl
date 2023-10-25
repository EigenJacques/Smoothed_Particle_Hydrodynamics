
function rectangle(X)

    # Domain boundaries 
    function upper(X, a)
        return -X[2] .+ a
    end
    
    function lower(X, a)
        return X[2] .+ a
    end
    
    function left(X, a)
        return X[1] .+ a
    end
    
    function right(X, a)
        return -X[1] .+ a
    end

    # R-function definition of domain
    n = 2

    x1 = upper(X,0.5)
    x2 = lower(X,0.5)
    x3 = left(X,0.5)
    x4 = right(X,0.5)
    x5 = (x1 .+ x2 .- (x1.^n .+ x2.^n ).^(1/n))
    x6 = (x3 .+ x4 .- (x3.^n .+ x4.^n ).^(1/n))

    return x5 .+ x6 .- (x5.^n .+ x6.^n).^(1/n)

end

using Plots

x = range(-1, 1, length=100)
y = range(-1, 1, length=100)

z = rectangle([x',y])

contour(x,y,z, levels=[0], color=:turbo, cbar=false, lw=1)

using BenchmarkTools
@benchmark rectangle([x',y])

