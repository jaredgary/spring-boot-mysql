# Spring Boot MySQL Example

See docker hub repositories from jaredgary

Steps to run single app
1. Run docker-compose form ./src/main/docker/mysql directory: docker-compose up -d
2. Run mvn clean install (need to hac=ve docker on localhost up)
3. Run docker-compose form ./src/main/docker/myapp directory: docker-compose up -d

Steps to run docker stack deploy (Swarm Cluster)
1. Run docker-compose form ./src/main/scripts/stack/mysql directory: docker stack deploy -c docker-compose.yml <name>
2. Upload image on docker hub repository
3. Run docker-compose form ./src/main/scripts/stack/myapp directory: docker stack deploy -c docker-compose.yml <name>

Finish