from typing import Dict, Any, List, Optional, Callable # Ensure Callable is imported
import logging
import numpy as np # Ensure numpy is imported
import asyncio # Ensure asyncio is imported
import random
import pickle # Ensure pickle is imported for the proxy if it needs to handle it directly
import base64 # Ensure base64 is imported for the proxy
import copy
import os
import threading
import time
import uuid

# Attempt to import httpx, or define a placeholder if not available
# In a real setup, httpx should be in requirements.txt
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None # type: ignore
    logger.warning("httpx library is not installed. RemoteAggregationServerProxy will not function.")
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Placeholder for a local model that would be used with FederatedNetworkOptimizer
class SampleLocalModel:
    def __init__(self):
        self.parameters = {}
        logger.info("SampleLocalModel initialized.")

    async def train(self, data: Any) -> Dict[str, Any]:
        logger.info(f"SampleLocalModel training with data of type: {type(data)}.")
        # Simulate training
        return {"update_type": "simulated_gradient", "value": [0.1, 0.2, 0.3]}

    def apply_parameters(self, params: Dict[str, Any]):
        logger.info(f"SampleLocalModel applying parameters: {params}.")
        self.parameters = params

# Placeholder for AggregationServer
class AggregationServer:
    def __init__(self):
        logger.info("AggregationServer initialized.")
        # Initialize with a named parameter, e.g., for a simple model
        self.global_model_parameters: Dict[str, np.ndarray] = {"default_weights": np.array([1.0, 1.0, 1.0])} 
        self.received_updates: List[Dict[str, Any]] = []
        self.current_round = 0
        self.global_model = None
        self.model_architecture = None
        self.client_updates = {}
        self.round_active = False
        self.lock = threading.Lock()
        self.client_counts = {}
        self.id = str(uuid.uuid4())[:8]
        logger.info(f"AggregationServer {self.id} initialized")

    async def start(self):
        logger.info("AggregationServer started for a new federation round.")
        self.current_round +=1
        self.received_updates = [] # Clear updates for the new round

    async def register_client(self, client_id: str):
        logger.info(f"Client {client_id} registered with AggregationServer.")
        # In a real system, might track active clients

    async def distribute_model_parameters(self) -> Dict[str, Any]: # Renamed for clarity
        logger.info(f"AggregationServer distributing global model parameters for round {self.current_round}: {self.global_model_parameters}")
        return self.global_model_parameters
    
    async def receive_update(self, client_id: str, local_update: Dict[str, Any], num_samples: int):
        logger.info(f"AggregationServer received update from client {client_id} (samples: {num_samples}): {local_update.get('update_type')}")
        self.received_updates.append({"client_id": client_id, "update": local_update, "num_samples": num_samples})
        # Potentially trigger aggregation if enough updates are received

    async def aggregate_model_updates(self) -> Dict[str, Any]:
        logger.info(f"AggregationServer aggregating {len(self.received_updates)} updates for round {self.current_round}.")
        if not self.received_updates:
            logger.warning("No updates received for aggregation. Global model remains unchanged.")
            return self.global_model_parameters

        # Determine structure from the first valid update that contains parameter data
        # Assumes updates provide parameters as a dictionary (key: param_name, value: np.array)
        # or a single np.array (which will be stored under a default key like 'default_weights')
        aggregated_params_dict: Dict[str, np.ndarray] = {}
        first_valid_update_value = None
        for update_info in self.received_updates:
            value = update_info["update"].get("value")
            if value is not None:
                first_valid_update_value = value
                break
        
        if first_valid_update_value is None:
            logger.warning("No parameter values found in any client updates. Global model remains unchanged.")
            return self.global_model_parameters

        if isinstance(first_valid_update_value, dict): # Model parameters are a dict of arrays
            param_keys = first_valid_update_value.keys()
            for key in param_keys:
                weighted_sum = np.zeros_like(first_valid_update_value[key], dtype=np.float64)
                contributing_samples = 0
                for update_info in self.received_updates:
                    client_params_dict = update_info["update"].get("value")
                    if isinstance(client_params_dict, dict) and key in client_params_dict:
                        param_val = np.array(client_params_dict[key])
                        num_samples = update_info["num_samples"]
                        if num_samples > 0:
                            weighted_sum += (param_val * num_samples)
                            contributing_samples += num_samples
                if contributing_samples > 0:
                    aggregated_params_dict[key] = weighted_sum / contributing_samples
                else:
                    # Fallback to existing global if no one contributed to this param key
                    if key in self.global_model_parameters:
                        aggregated_params_dict[key] = self.global_model_parameters[key]
                        logger.warning(f"No updates for parameter '{key}'. Retaining previous global value.")

        elif isinstance(first_valid_update_value, (list, np.ndarray)): # Model parameters are a single list/array
            param_len = len(first_valid_update_value)
            weighted_sum_list = np.zeros(param_len, dtype=np.float64)
            contributing_samples = 0
            for update_info in self.received_updates:
                client_params_list = update_info["update"].get("value")
                num_samples = update_info["num_samples"]
                if isinstance(client_params_list, (list, np.ndarray)) and len(client_params_list) == param_len and num_samples > 0:
                    weighted_sum_list += (np.array(client_params_list) * num_samples)
                    contributing_samples += num_samples
            if contributing_samples > 0:
                # Store this single array under a default key, e.g., 'default_weights'
                # This key should match how it's initialized or expected by models.
                # For consistency, let's use the first key from initial global_model_parameters if available,
                # or a hardcoded default.
                default_key = next(iter(self.global_model_parameters.keys()), "default_weights")
                aggregated_params_dict[default_key] = weighted_sum_list / contributing_samples
            else:
                logger.warning("No valid list/array updates contributed. Retaining previous global model values for array-based params.")
                # Try to preserve existing structure if possible
                # This part is tricky if the initial model was dict and updates are list.
                # For now, if this path is hit without updates, it might lead to empty aggregated_params_dict for this type.
                pass 

        if aggregated_params_dict: 
             self.global_model_parameters = aggregated_params_dict
             logger.info(f"Aggregation complete. New global model parameters: {self.global_model_parameters}")
        else:
            logger.warning("Aggregation resulted in empty parameters or no valid updates. Global model not updated.")
        
        self.received_updates = [] # Clear updates after aggregation for this round
        return self.global_model_parameters

class RemoteAggregationServerProxy:
    """Acts as a local proxy for a remote AggregationServer, communicating via HTTP."""
    def __init__(self, coordinator_base_url: str, client_id: str, http_client: Optional[httpx.AsyncClient] = None):
        if httpx is None:
            raise ImportError("httpx is required for RemoteAggregationServerProxy but not installed.")
        self.coordinator_base_url = coordinator_base_url.rstrip('/')
        self.client_id = client_id # Needed for context in some calls
        self.http_client = http_client if http_client else httpx.AsyncClient()
        # Note: For a real app, manage the lifecycle of http_client (e.g., close it)
        self.id = str(uuid.uuid4())[:8]
        logger.info(f"RemoteAggregationServerProxy {self.id} initialized with URL: {coordinator_base_url}")

    async def register_client(self, client_id: str):
        # This is a conceptual registration. The actual API might vary.
        # For now, assume registration happens implicitly or via a separate mechanism.
        logger.info(f"Proxy: Client {client_id} registration with remote server is conceptual.")
        # Example: await self.http_client.post(f"{self.coordinator_base_url}/api/federated-learning/register-client", json={"client_id": client_id})
        # No specific register_client endpoint in current routes, depends on server design.
        pass 

    async def distribute_model_parameters(self) -> Dict[str, Any]:
        logger.info(f"Proxy ({self.client_id}): Requesting global model from {self.coordinator_base_url}/api/federated-learning/global-model")
        try:
            response = await self.http_client.get(f"{self.coordinator_base_url}/api/federated-learning/global-model")
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            if data.get("format") == "pickle_base64":
                pickled_model = base64.b64decode(data["model_b64"])
                model_params = pickle.loads(pickled_model)
                logger.info(f"Proxy ({self.client_id}): Successfully received and deserialized global model.")
                return model_params
            else:
                logger.error(f"Proxy ({self.client_id}): Unknown model format received: {data.get('format')}")
                raise ValueError("Unknown model format received from server")
        except httpx.RequestError as e:
            logger.error(f"Proxy ({self.client_id}): HTTP request error fetching global model: {e}")
            raise # Re-raise to allow handling by the client
        except (pickle.PickleError, base64.binascii.Error, KeyError, ValueError) as e:
            logger.error(f"Proxy ({self.client_id}): Error processing global model data: {e}")
            raise

    async def receive_update(self, client_id: str, local_update: Dict[str, Any], num_samples: int):
        logger.info(f"Proxy ({client_id}): Submitting local update ({num_samples} samples) to {self.coordinator_base_url}/api/federated-learning/client-update")
        try:
            pickled_update = pickle.dumps(local_update)
            update_b64 = base64.b64encode(pickled_update).decode('utf-8')
            
            payload = {"update_b64": update_b64, "format": "pickle_base64"}
            # The client_id and num_samples are part of the route path/query in the API definition
            # So they need to be passed in the URL, not the JSON body as per current API.
            # Let's adjust the API call to match the FastAPI endpoint definition.
            # The API has: client_id: str, num_samples: int as separate params, then update_data (body)
            # This means they should be query parameters or path parameters.
            # For simplicity, let's assume they are query parameters for now.
            # This is a slight mismatch with how AggregationServer.receive_update is called internally by FederatedClient.
            # The call to the API should be to /client-update?client_id=...&num_samples=...
            # The body is `update_data` which contains `update_b64` and `format`.

            api_url = f"{self.coordinator_base_url}/api/federated-learning/client-update?client_id={client_id}&num_samples={num_samples}"

            response = await self.http_client.post(api_url, json=payload)
            response.raise_for_status()
            logger.info(f"Proxy ({client_id}): Successfully submitted local update.")
            # response_data = response.json() # if server sends back useful info
            # return response_data
            return # Typically POST doesn't return much beyond status
        except httpx.RequestError as e:
            logger.error(f"Proxy ({client_id}): HTTP request error submitting update: {e}")
            raise
        except (pickle.PickleError, base64.binascii.Error) as e:
            logger.error(f"Proxy ({client_id}): Error serializing local update: {e}")
            raise

# Placeholder for FederatedClient
class FederatedClient:
    def __init__(self, client_id: str, server_interface: Any, local_data_source: Any):
        self.client_id = client_id
        # server_interface can be an AggregationServer instance or RemoteAggregationServerProxy instance
        self.server_interface = server_interface 
        self.local_data_source = local_data_source
        self.local_model = SampleLocalModel()
        logger.info(f"FederatedClient {client_id} initialized.")
        # Registration might be handled differently if remote
        if isinstance(server_interface, AggregationServer): # Direct local server
            asyncio.create_task(self.server_interface.register_client(self.client_id))
        elif isinstance(server_interface, RemoteAggregationServerProxy):
            # Proxy might have its own registration logic or it's handled at a higher level
            asyncio.create_task(self.server_interface.register_client(self.client_id))
            logger.info(f"Client {client_id} will communicate with remote server via proxy.")

    async def get_global_model_parameters(self) -> Dict[str, Any]:
        logger.info(f"Client {self.client_id} requesting global model parameters.")
        return await self.server_interface.distribute_model_parameters()

    async def submit_local_update(self, local_update: Dict[str, Any], num_samples: int):
        logger.info(f"Client {self.client_id} submitting local update based on {num_samples} samples.")
        await self.server_interface.receive_update(self.client_id, local_update, num_samples)
    
    async def run_training_round(self):
        logger.info(f"Client {self.client_id} starting training round.")
        # 1. Get global model parameters
        global_params = await self.get_global_model_parameters()
        self.local_model.apply_parameters(global_params)
        logger.info(f"Client {self.client_id} applied global parameters.")

        # 2. Train local model on local data
        #    In a real scenario, local_data would be fetched and preprocessed.
        local_data, num_samples = self.local_data_source.get_anonymized_network_data() # Assume this method exists
        if num_samples == 0:
            logger.warning(f"Client {self.client_id}: No local data available for this round. Skipping update.")
            return

        logger.info(f"Client {self.client_id} training local model on {num_samples} samples.")
        local_update_payload = await self.local_model.train(local_data)
        
        # 3. Submit local update to server
        await self.submit_local_update(local_update_payload, num_samples)
        logger.info(f"Client {self.client_id} submitted local update to server.")

class FederatedNetworkOptimizer:
    """Enables collaborative learning across multiple data centers while preserving privacy"""
    
    def __init__(self, local_model_creator: Callable[[], Any] , aggregation_server: Optional[AggregationServer] = None, coordinator_api_url: Optional[str] = None):
        """
        Args:
            local_model_creator: A function that creates an instance of a local model.
            aggregation_server: An instance of AggregationServer if this node is the coordinator (direct mode).
            coordinator_api_url: Base URL of the coordinator API if this node is a client connecting remotely.
        """
        self.local_model_creator = local_model_creator
        self.aggregation_server = aggregation_server # This is for the coordinator role or local client mode
        self.coordinator_api_url = coordinator_api_url # For remote client mode
        self.clients: List[FederatedClient] = []
        self.is_coordinator = bool(aggregation_server) and not bool(coordinator_api_url)
        self.client_node: Optional[FederatedClient] = None
        
    async def initialize_federation(self, client_id: str = "client_default", local_data_source: Any = None, mode: str = "auto"):
        """Set up federated learning infrastructure or client participation.
        Args:
            client_id: ID for this client if in client mode.
            local_data_source: Data source for this client if in client mode.
            mode: 'coordinator', 'local_client', or 'remote_client'. 'auto' determines from init args.
        """
        effective_mode = mode
        if mode == "auto":
            if self.is_coordinator:
                effective_mode = "coordinator"
            elif self.coordinator_api_url and local_data_source: # Indicates remote client potential
                effective_mode = "remote_client"
            elif self.aggregation_server and local_data_source: # Local server instance provided for client
                 effective_mode = "local_client"
            else:
                logger.error("Initialization error: Cannot determine mode. Provide server or API URL for client, or server for coordinator.")
                return

        if effective_mode == "coordinator" and self.aggregation_server:
            await self.aggregation_server.start()
            logger.info("Federation Coordinator initialized and server started.")
        elif effective_mode == "remote_client" and self.coordinator_api_url and local_data_source:
            if httpx is None:
                logger.error("Cannot initialize remote client: httpx is not installed.")
                return
            proxy = RemoteAggregationServerProxy(coordinator_base_url=self.coordinator_api_url, client_id=client_id)
            self.client_node = FederatedClient(client_id, proxy, local_data_source)
            logger.info(f"Federated Remote Client Node {client_id} initialized, connecting to {self.coordinator_api_url}.")
        elif effective_mode == "local_client" and self.aggregation_server and local_data_source:
            self.client_node = FederatedClient(client_id, self.aggregation_server, local_data_source)
            logger.info(f"Federated Local Client Node {client_id} initialized, using direct server connection.")
        else:
            logger.error(f"Initialization error for mode '{effective_mode}': Invalid setup or missing components.")
           
    # Methods for Coordinator Role
    def add_client_node(self, client_id: str, local_data_source: Any):
        if not self.is_coordinator or not self.aggregation_server:
            logger.error("Cannot add client: This node is not a coordinator or server not initialized.")
            return
        client = FederatedClient(client_id, self.aggregation_server, local_data_source)
        self.clients.append(client)
        logger.info(f"Client {client_id} added to federation by coordinator.")

    async def run_federation_round(self):
        if not self.is_coordinator or not self.aggregation_server:
            logger.error("Cannot run federation round: Not a coordinator or server not initialized.")
            return None

        logger.info(f"Coordinator starting federation round {self.aggregation_server.current_round + 1}.")
        await self.aggregation_server.start() # Prepares server for new round (e.g. clears old updates)

        # 1. Distribute global model (implicitly done when clients request)
        # global_params = await self.aggregation_server.distribute_model_parameters()
        # (Clients will fetch this themselves)

        # 2. Trigger clients to train and send updates
        client_training_tasks = [client.run_training_round() for client in self.clients]
        await asyncio.gather(*client_training_tasks)
        logger.info("Coordinator: All clients completed local training round.")

        # 3. Aggregate updates
        new_global_model_params = await self.aggregation_server.aggregate_model_updates()
        logger.info(f"Coordinator: Aggregation complete. New global model: {new_global_model_params}")
        return new_global_model_params

    # Method for Client Role (if this optimizer instance itself is a client)
    async def participate_as_client(self):
        if self.client_node:
            await self.client_node.run_training_round()
        else:
            logger.error("Client participation failed: Optimizer not initialized as a client node.")

    # These methods are deprecated if using the client/server role structure above
    # async def train_local_model(self, local_data: Any) -> Dict[str, Any]: ...
    # async def participate_in_round(self): ...
    # async def aggregate_updates(self, updates): ...

    def get_anonymized_network_data(self) -> Tuple[Dict[str, Any], int]: # Now returns num_samples too
        """Placeholder for getting anonymized local network data and sample count."""
        logger.info("Getting anonymized network data.")
        data = {"feature1": np.random.rand(100), "feature2": np.random.rand(100), "target": np.random.randint(0, 2, 100)}
        return data, len(data["target"])

# Example Usage (Illustrative - would typically be in different parts of a larger system)
async def main_fl_example():
    # --- Scenario 1: Coordinator and local clients (original example, slightly adapted) ---
    print("\n--- SCENARIO 1: Coordinator with Local Clients ---")
    coordinator_server_local = AggregationServer()
    federation_coordinator_local = FederatedNetworkOptimizer(
        local_model_creator=lambda: SampleLocalModel(), 
        aggregation_server=coordinator_server_local
    )
    await federation_coordinator_local.initialize_federation(mode="coordinator")

    class DummyDataSourceLocal:
        def __init__(self, client_name: str):
            self.client_name = client_name
        def get_anonymized_network_data(self):
            num_items = random.randint(50,150)
            data = {
                "feature1": np.random.rand(num_items),
                "feature2": np.random.rand(num_items),
                # Parameters are expected as dict by refined AggregationServer
                "value": {"default_weights": np.random.rand(3)} 
            }
            # SampleLocalModel.train expects data that it can process, not necessarily features/target
            # For this example, let's assume SampleLocalModel can directly use this data dict if needed,
            # or it ignores it and produces a generic update. The key is num_samples.
            logger.debug(f"DataSource for {self.client_name} providing {num_items} samples.")
            return data, num_items
    
    dataSource1_local = DummyDataSourceLocal("client_A_local")
    dataSource2_local = DummyDataSourceLocal("client_B_local")

    federation_coordinator_local.add_client_node("client_A_local", dataSource1_local)
    federation_coordinator_local.add_client_node("client_B_local", dataSource2_local)

    for round_num_local in range(2): # Reduced rounds for brevity
        print(f"\n--- Starting Local FL Round {round_num_local + 1} ---")
        updated_global_model_local = await federation_coordinator_local.run_federation_round()
        if updated_global_model_local:
            print(f"Local global model updated in round {round_num_local + 1}: {updated_global_model_local}")

    # --- Scenario 2: Remote Client connecting to a (mocked) remote coordinator ---
    # This part is conceptual as it requires a running server with the FastAPI routes.
    # We can simulate the client part.
    print("\n\n--- SCENARIO 2: Remote Client (Conceptual) ---")
    # Assume a coordinator is running at http://localhost:8000 (where FastAPI app would be)
    mock_coordinator_url = "http://mock-coordinator-url.com" # Placeholder URL
    
    # Client Node Setup for Remote Connection
    remote_client_optimizer = FederatedNetworkOptimizer(
        local_model_creator=lambda: SampleLocalModel(), 
        coordinator_api_url=mock_coordinator_url
    )
    dataSource_remote = DummyDataSourceLocal("client_C_remote")
    await remote_client_optimizer.initialize_federation(
        client_id="client_C_remote", 
        local_data_source=dataSource_remote, 
        mode="remote_client"
    )

    if remote_client_optimizer.client_node and httpx: # Check if client node created and httpx available
        print(f"Remote client {remote_client_optimizer.client_node.client_id} initialized.")
        print("Conceptual: This client would now participate in rounds by communicating with the coordinator API.")
        print("To run this for real, point mock_coordinator_url to a running instance of the FastAPI app.")
        # Example of how a client might participate (conceptual call, needs running server to test)
        # try:
        #     print("Attempting to participate in a remote round (conceptual)...")
        #     await remote_client_optimizer.participate_as_client()
        #     print("Remote client participation call completed (check server logs for actual interaction if server was live).")
        # except Exception as e:
        #     print(f"Error during conceptual remote participation: {e}")
    else:
        if not httpx:
            print("Skipping remote client scenario: httpx not installed.")
        else:
            print("Skipping remote client scenario: client node not initialized correctly.")

if __name__ == "__main__":
    asyncio.run(main_fl_example())