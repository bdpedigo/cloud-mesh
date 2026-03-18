# comment
gcloud config set project exalted-beanbag-334502

# machine-type
# to see list of machines, do:
# $ gcloud compute machine-types list --filter="zone:(us-east4-b)"

# disk-size: Size for node VM boot disks in GB. Defaults to 100GB.

# metadata: Compute Engine metadata to be made available to the guest operating
# system running on nodes within the node pool.

# num-nodes: The number of nodes to be created in each of the cluster's zones.
# --machine-type "c2d-standard-32" \

CORE_COUNT=32
NUM_NODES=8
MACHINE_TYPE="c2d-highmem-${CORE_COUNT}"
CLUSTER_NAME="cloud-mesh-${CORE_COUNT}x${NUM_NODES}"

# gcloud container --project "exalted-beanbag-334502" clusters delete "cloud-mesh" --zone "us-west1-b"
gcloud container --project "exalted-beanbag-334502" clusters create "${CLUSTER_NAME}" \
    --zone "us-west1-c" \
    --no-enable-basic-auth \
    --release-channel "stable" \
    --machine-type "${MACHINE_TYPE}" \
    --image-type "COS_CONTAINERD" \
    --disk-type "pd-standard" \
    --disk-size "600" \
    --metadata disable-legacy-endpoints=true \
    --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
    --preemptible \
    --num-nodes "${NUM_NODES}" \
    --logging=SYSTEM,WORKLOAD \
    --monitoring=SYSTEM \
    --enable-ip-alias \
    --network "projects/exalted-beanbag-334502/global/networks/patchseq" \
    --subnetwork "projects/exalted-beanbag-334502/regions/us-west1/subnetworks/patchseq" \
    --no-enable-intra-node-visibility \
    --no-enable-master-authorized-networks \
    --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
    --enable-autoupgrade \
    --enable-autorepair \
    --max-unavailable-upgrade 0 \
    --max-pods-per-node "256" \
    --node-locations "us-west1-c" \
    --enable-shielded-nodes \
    --shielded-secure-boot \
    --shielded-integrity-monitoring

gcloud container clusters get-credentials --zone us-west1-c "${CLUSTER_NAME}"

# https://kubernetes.io/docs/concepts/configuration/secret/
kubectl create secret generic secrets \
    --from-file=$HOME/.cloudvolume/secrets/cave-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/aws-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/google-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/discord-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/cave_datastack_to_server_map.json \
    --from-file=$HOME/.cloudvolume/secrets/globalv1.em.brain.allentech.org-cave-secret.json \
    --from-file=$HOME/.cloudvolume/secrets/global.em.brain.allentech.org-cave-secret.json \

kubectl apply -f kube-task.yml