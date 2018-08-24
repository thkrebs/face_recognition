gcloud container clusters create cluster-1 --num-nodes 1  --machine-type n1-standard-1
gcloud container clusters get-credentials cluster-1
kubectl apply -f fr_deployment.yml