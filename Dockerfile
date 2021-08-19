FROM julia:latest

COPY ./src /app/src
COPY ./util /app/util
COPY ./main.jl /app
COPY ./Project.toml /app
# COPY ./Manifest.toml /app

# COPY ./WaspNet.jl /app/WaspNet.jl
# RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path="/app/WaspNet.jl")'

#CMD ["julia", "./app/main.jl", "-m", "NN", "-e", "100"]
# RUN cd app

#RUN julia --project main.jl -m NN -e 100
