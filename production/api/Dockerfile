FROM maven:3.9.9-eclipse-temurin-17 AS build
WORKDIR /build

COPY pom.xml ./pom.xml
COPY src ./src

RUN mvn -B -DskipTests clean package

FROM eclipse-temurin:17-jre
WORKDIR /app

COPY --from=build /build/target/ecg-front-0.0.1-SNAPSHOT.jar /app/app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
