# Expose the deployment
kubectl expose deployment fr --type=LoadBalancer --name=fr-svc
kubectl get service fr-svc