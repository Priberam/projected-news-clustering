FROM mcr.microsoft.com/dotnet/core/sdk:3.1-alpine3.10 AS build
WORKDIR /app
COPY ./DistilEvaluation /app/DistilEvaluation
COPY ./DistilMonoClustering /app/DistilMonoClustering
COPY ./models /app/models
WORKDIR /app/DistilEvaluation
RUN dotnet publish -r linux-musl-x64 -c Release -o ./deploy/release
RUN dotnet publish -r linux-musl-x64 -c Debug -o ./deploy/debug

FROM mcr.microsoft.com/dotnet/core/runtime-deps:3.1-alpine3.10
COPY --from=build /app/DistilEvaluation/deploy ./app
COPY --from=build /app/models ./app/models
COPY --from=build /app/DistilEvaluation/nlog.config ./app/release
COPY --from=build /app/DistilEvaluation/appsettings.json ./app/release
COPY --from=build /app/DistilEvaluation/nlog.config ./app/debug
RUN mkdir -p /app/state
RUN mkdir -p /app/out
RUN mkdir -p /app/license
COPY ./datasets /app/datasets
COPY LICENSE.txt /app/license

WORKDIR /app/release
ENTRYPOINT ["./DistilEvaluation"]