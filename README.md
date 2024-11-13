# Extracting features and labels

To build (from folder above):
`docker buildx build --platform linux/amd64 -t cloud-mesh . -f ./cloud-mesh/Dockerfile`

To run:
`docker run --rm --platform linux/amd64 -v /Users/ben.pedigo/.cloudvolume/secrets:/root/.cloudvolume/secrets cloud-mesh`

To tag:
`docker tag cloud-mesh bdpedigo/cloud-mesh:v0`

To push:
`docker push bdpedigo/cloud-mesh:v0`

Making a cluster:
`sh ./make_cluster.sh`

Configuring a cluster:
`kubectl apply -f kube-task.yml`

Monitor the cluster:
`kubectl get pods`

Watch the logs in real-time:
`kubectl logs -f <pod-name>`

Check for issues with the cluster:
`kubectl describe nodes`
