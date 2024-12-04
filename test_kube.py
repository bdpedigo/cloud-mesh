#%%
from kubernetes import client, config


def get_replicas_on_node(namespace, pod_label_selector):
    """
    Count the number of pod replicas running on a specific node in a Kubernetes cluster.

    :param namespace: The Kubernetes namespace to query.
    :param pod_label_selector: Label selector to filter the pods (e.g., 'app=my-app').
    :return: The number of running pod replicas on the specified node.
    """
    # Load Kubernetes config (from inside the cluster)
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        # running on my Mac
        return 1

    # Create a Kubernetes API client
    v1 = client.CoreV1Api()

    # List all pods in the namespace with the given label selector
    pods = v1.list_namespaced_pod(
        namespace=namespace, label_selector=pod_label_selector
    )

    # Filter pods that are running on the specified node and in 'Running' state
    running_replicas = [pod for pod in pods.items if pod.status.phase == "Running"]

    return len(running_replicas)


# Example usage
namespace = "cloud-mesh"  # Change as needed
pod_label_selector = "cloud-mesh"  # Replace with your pod's label selector

replica_count = get_replicas_on_node(namespace, pod_label_selector)
print(f"Number of running replicas on node: {replica_count}")

# %%
