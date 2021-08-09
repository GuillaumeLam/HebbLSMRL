using Flux
using Zygote

x = rand(5)

A = rand(2,5)
b = rand(2)

struct Linear
    W
    b
end

(l::Linear)(x) = l.W * x .+ l.b

Flux.trainable(l::Linear) = (l.W,)

model = Linear(A, b)

dmodel = Zygote.gradient(() -> sum(model(x)), params(model))

dmodel.grads

for (k,v) in dmodel.grads
    println(typeof(k))
end
