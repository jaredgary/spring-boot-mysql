FROM openjdk:8-jdk-alpine
VOLUME /tmp
COPY ./target/spring-boot-mysql-1.0.0.jar spring-boot-mysql-1.0.0.jar
ENTRYPOINT ["java","-Djava.security.egd=file:/dev/./urandom","-jar","/spring-boot-mysql-1.0.0.jar"]