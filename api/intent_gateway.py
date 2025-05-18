"""
Intent-based API Gateway Module

This module processes natural language intent from users and maps it to API calls.
It provides a more intuitive interface for controlling and querying the optical network.
"""

import logging
import json
import re
import datetime
from typing import Dict, Any, List, Callable, Optional, Tuple
import asyncio
import uuid
import os

logger = logging.getLogger(__name__)

class LLMProcessor:
    """Processes natural language using LLM models to extract intent"""
    
    def __init__(self, model_name: str = "gpt-4", api_key: str = None):
        """Initialize the LLM processor
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Load prompt templates
        self.prompt_templates = {
            "intent_extraction": """
            Extract the user's intent from the following query. Identify:
            1. The primary action (e.g., get, create, update, delete)
            2. The target resource (e.g., network, device, link, simulation)
            3. Any parameters or filters
            4. Any temporal constraints (e.g., "last 24 hours")
            
            User query: {query}
            
            Return your response as a JSON object with the following structure:
            {{
                "action": string,
                "resource": string,
                "parameters": object,
                "temporal_constraints": object
            }}
            """,
            
            "api_mapping": """
            Given the extracted intent and available API endpoints, determine the most appropriate API call.
            
            Extracted intent: {intent}
            
            Available endpoints:
            {endpoints}
            
            Return your response as a JSON object with the following structure:
            {{
                "endpoint": string,
                "method": string,
                "parameters": object,
                "confidence": number
            }}
            """
        }
        
        logger.info(f"LLMProcessor initialized with model: {model_name}")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query to extract intent
        
        Args:
            query: Natural language query
            
        Returns:
            Extracted intent
        """
        logger.info(f"Processing query: {query}")
        
        # For demonstration purposes, we're using a pattern-matching approach
        # In a real implementation, this would call an LLM API
        
        intent = self._extract_intent_demo(query)
        logger.info(f"Extracted intent: {json.dumps(intent)}")
        
        return intent
    
    def _extract_intent_demo(self, query: str) -> Dict[str, Any]:
        """Extract intent from query using pattern matching (demo version)
        
        Args:
            query: Natural language query
            
        Returns:
            Extracted intent
        """
        # Simple pattern matching for demonstration
        query = query.lower()
        
        # Initialize intent structure
        intent = {
            "action": None,
            "resource": None,
            "parameters": {},
            "temporal_constraints": {}
        }
        
        # Extract action
        if any(word in query for word in ["show", "get", "list", "display", "what"]):
            intent["action"] = "get"
        elif any(word in query for word in ["create", "add", "new", "setup"]):
            intent["action"] = "create"
        elif any(word in query for word in ["update", "change", "modify", "adjust"]):
            intent["action"] = "update"
        elif any(word in query for word in ["delete", "remove", "destroy"]):
            intent["action"] = "delete"
        elif any(word in query for word in ["simulate", "run simulation"]):
            intent["action"] = "simulate"
        else:
            intent["action"] = "get"  # Default to get
        
        # Extract resource
        resources = ["network", "device", "link", "simulation", "component", "transceiver", "switch", "topology"]
        for resource in resources:
            if resource in query:
                intent["resource"] = resource
                break
        
        # Extract parameters
        # ID parameter
        id_match = re.search(r"id (\w+)", query)
        if id_match:
            intent["parameters"]["id"] = id_match.group(1)
        
        # Name parameter
        name_match = re.search(r"named (\w+)", query)
        if name_match:
            intent["parameters"]["name"] = name_match.group(1)
        
        # Status parameter
        if "active" in query:
            intent["parameters"]["status"] = "active"
        elif "inactive" in query:
            intent["parameters"]["status"] = "inactive"
        
        # Extract temporal constraints
        if "last 24 hours" in query or "today" in query:
            intent["temporal_constraints"] = {
                "start_time": (datetime.datetime.now() - datetime.timedelta(hours=24)).isoformat(),
                "end_time": datetime.datetime.now().isoformat()
            }
        elif "last week" in query:
            intent["temporal_constraints"] = {
                "start_time": (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat(),
                "end_time": datetime.datetime.now().isoformat()
            }
        elif "last month" in query:
            intent["temporal_constraints"] = {
                "start_time": (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
                "end_time": datetime.datetime.now().isoformat()
            }
        
        return intent
    
    async def map_to_api_call(self, intent: Dict[str, Any], api_registry: 'APIRegistry') -> Dict[str, Any]:
        """Map extracted intent to an API call
        
        Args:
            intent: Extracted intent
            api_registry: Registry of available API endpoints
            
        Returns:
            API call details
        """
        logger.info(f"Mapping intent to API call: {json.dumps(intent)}")
        
        # Get matching endpoints from the registry
        matching_endpoints = api_registry.find_matching_endpoints(
            action=intent["action"],
            resource=intent["resource"]
        )
        
        if not matching_endpoints:
            return {
                "error": "No matching API endpoint found",
                "intent": intent
            }
        
        # Select the best endpoint (in a real implementation, this would use the LLM)
        selected_endpoint = matching_endpoints[0]
        
        # Map intent parameters to API parameters
        api_parameters = self._map_parameters(intent["parameters"], selected_endpoint["parameters"])
        
        # Add temporal constraints if applicable
        if intent["temporal_constraints"] and "temporal_parameters" in selected_endpoint:
            for temp_param, api_param in selected_endpoint["temporal_parameters"].items():
                if temp_param in intent["temporal_constraints"]:
                    api_parameters[api_param] = intent["temporal_constraints"][temp_param]
        
        return {
            "endpoint": selected_endpoint["path"],
            "method": selected_endpoint["method"],
            "parameters": api_parameters,
            "confidence": 0.9  # Placeholder confidence score
        }
    
    def _map_parameters(self, intent_params: Dict[str, Any], endpoint_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map intent parameters to API parameters
        
        Args:
            intent_params: Parameters from the intent
            endpoint_params: Parameters expected by the endpoint
            
        Returns:
            Mapped API parameters
        """
        api_parameters = {}
        
        for api_param, param_info in endpoint_params.items():
            intent_param = param_info.get("intent_param", api_param)
            if intent_param in intent_params:
                api_parameters[api_param] = intent_params[intent_param]
            elif param_info.get("required", False):
                # For required parameters, provide a default if possible
                if "default" in param_info:
                    api_parameters[api_param] = param_info["default"]
        
        return api_parameters

class APIRegistry:
    """Registry of available API endpoints"""
    
    def __init__(self):
        """Initialize the API registry"""
        self.endpoints = []
        self._load_endpoints()
        logger.info(f"APIRegistry initialized with {len(self.endpoints)} endpoints")
    
    def _load_endpoints(self):
        """Load API endpoints from configuration"""
        # In a real implementation, this would load from a configuration file or database
        # For now, we'll add some example endpoints
        
        self.endpoints = [
            {
                "path": "/api/networks",
                "method": "GET",
                "description": "Get all networks",
                "action": "get",
                "resource": "network",
                "parameters": {}
            },
            {
                "path": "/api/networks/{id}",
                "method": "GET",
                "description": "Get a specific network by ID",
                "action": "get",
                "resource": "network",
                "parameters": {
                    "id": {
                        "type": "string",
                        "required": True,
                        "intent_param": "id"
                    }
                }
            },
            {
                "path": "/api/networks",
                "method": "POST",
                "description": "Create a new network",
                "action": "create",
                "resource": "network",
                "parameters": {
                    "name": {
                        "type": "string",
                        "required": True,
                        "intent_param": "name"
                    },
                    "description": {
                        "type": "string",
                        "required": False,
                        "intent_param": "description"
                    },
                    "topology_type": {
                        "type": "string",
                        "required": False,
                        "intent_param": "topology",
                        "default": "mesh"
                    }
                }
            },
            {
                "path": "/api/devices",
                "method": "GET",
                "description": "Get all devices",
                "action": "get",
                "resource": "device",
                "parameters": {
                    "status": {
                        "type": "string",
                        "required": False,
                        "intent_param": "status"
                    }
                },
                "temporal_parameters": {
                    "start_time": "created_after",
                    "end_time": "created_before"
                }
            },
            {
                "path": "/api/simulations",
                "method": "POST",
                "description": "Run a simulation",
                "action": "simulate",
                "resource": "simulation",
                "parameters": {
                    "network_id": {
                        "type": "string",
                        "required": True,
                        "intent_param": "id"
                    },
                    "duration": {
                        "type": "number",
                        "required": False,
                        "intent_param": "duration",
                        "default": 3600
                    },
                    "traffic_pattern": {
                        "type": "string",
                        "required": False,
                        "intent_param": "pattern",
                        "default": "mixed"
                    }
                }
            }
        ]
    
    def find_matching_endpoints(self, action: str, resource: str) -> List[Dict[str, Any]]:
        """Find endpoints matching the given action and resource
        
        Args:
            action: Action to perform (get, create, update, delete)
            resource: Resource to act upon
            
        Returns:
            List of matching endpoints
        """
        return [
            endpoint for endpoint in self.endpoints
            if endpoint["action"] == action and endpoint["resource"] == resource
        ]
    
    def get_all_endpoints(self) -> List[Dict[str, Any]]:
        """Get all registered endpoints
        
        Returns:
            List of all endpoints
        """
        return self.endpoints
    
    def register_endpoint(self, endpoint: Dict[str, Any]):
        """Register a new endpoint
        
        Args:
            endpoint: Endpoint details
        """
        self.endpoints.append(endpoint)
        logger.info(f"Registered new endpoint: {endpoint['path']} ({endpoint['method']})")

class Validator:
    """Validates API requests and responses"""
    
    def __init__(self):
        """Initialize the validator"""
        self.schemas = {}
        self._load_schemas()
        logger.info("Validator initialized")
    
    def _load_schemas(self):
        """Load JSON schemas for validation"""
        # In a real implementation, this would load from schema files
        # For now, we'll add some example schemas
        
        self.schemas = {
            "network": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "topology_type": {"type": "string", "enum": ["mesh", "star", "ring", "bus"]}
                }
            },
            "device": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "location": {"type": "string"},
                    "status": {"type": "string", "enum": ["active", "inactive", "maintenance"]}
                }
            },
            "simulation": {
                "type": "object",
                "required": ["network_id"],
                "properties": {
                    "network_id": {"type": "string"},
                    "duration": {"type": "number", "minimum": 0},
                    "traffic_pattern": {"type": "string"}
                }
            }
        }
    
    def validate_request(self, resource: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a request against its schema
        
        Args:
            resource: Resource type
            data: Request data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if resource not in self.schemas:
            return True, None  # No schema to validate against
        
        schema = self.schemas[resource]
        
        # Check required fields
        for field in schema.get("required", []):
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check property types
        for field, value in data.items():
            if field in schema.get("properties", {}):
                prop_schema = schema["properties"][field]
                
                # Type check
                if prop_schema.get("type") == "string" and not isinstance(value, str):
                    return False, f"Field {field} must be a string"
                elif prop_schema.get("type") == "number" and not isinstance(value, (int, float)):
                    return False, f"Field {field} must be a number"
                
                # Enum check
                if "enum" in prop_schema and value not in prop_schema["enum"]:
                    return False, f"Field {field} must be one of: {', '.join(prop_schema['enum'])}"
                
                # Minimum value check
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    return False, f"Field {field} must be at least {prop_schema['minimum']}"
        
        return True, None

class IntentGateway:
    """Gateway that processes natural language intent and maps to API calls"""
    
    def __init__(self):
        """Initialize the intent gateway"""
        self.llm_processor = LLMProcessor()
        self.api_registry = APIRegistry()
        self.validator = Validator()
        self.request_history = []
        logger.info("IntentGateway initialized")
    
    async def process_intent(self, query: str) -> Dict[str, Any]:
        """Process a natural language intent and execute the corresponding API call
        
        Args:
            query: Natural language query
            
        Returns:
            API response
        """
        logger.info(f"Processing intent: {query}")
        
        # Extract intent from the query
        intent = await self.llm_processor.process_query(query)
        
        # Map intent to API call
        api_call = await self.llm_processor.map_to_api_call(intent, self.api_registry)
        
        if "error" in api_call:
            return {
                "status": "error",
                "message": api_call["error"],
                "intent": intent
            }
        
        # Validate the API call parameters
        resource = intent["resource"]
        is_valid, error = self.validator.validate_request(resource, api_call["parameters"])
        
        if not is_valid:
            return {
                "status": "error",
                "message": f"Validation error: {error}",
                "intent": intent,
                "api_call": api_call
            }
        
        # Record the request
        request_id = str(uuid.uuid4())
        self.request_history.append({
            "id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "intent": intent,
            "api_call": api_call
        })
        
        # In a real implementation, this would execute the API call
        # For now, we'll return a simulated response
        
        return {
            "status": "success",
            "request_id": request_id,
            "intent": intent,
            "api_call": api_call,
            "response": self._simulate_api_response(api_call)
        }
    
    def _simulate_api_response(self, api_call: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an API response for demonstration purposes
        
        Args:
            api_call: API call details
            
        Returns:
            Simulated API response
        """
        endpoint = api_call["endpoint"]
        method = api_call["method"]
        
        if endpoint == "/api/networks" and method == "GET":
            return {
                "networks": [
                    {"id": "net1", "name": "Data Center Network", "topology_type": "mesh"},
                    {"id": "net2", "name": "Campus Network", "topology_type": "star"},
                    {"id": "net3", "name": "Backbone Network", "topology_type": "ring"}
                ]
            }
        elif endpoint.startswith("/api/networks/") and method == "GET":
            network_id = api_call["parameters"].get("id", "unknown")
            return {
                "id": network_id,
                "name": f"Network {network_id}",
                "topology_type": "mesh",
                "devices": [
                    {"id": "dev1", "name": "Device 1", "type": "switch"},
                    {"id": "dev2", "name": "Device 2", "type": "router"}
                ]
            }
        elif endpoint == "/api/networks" and method == "POST":
            return {
                "id": "new_net_" + str(uuid.uuid4())[:8],
                "name": api_call["parameters"].get("name", "New Network"),
                "topology_type": api_call["parameters"].get("topology_type", "mesh"),
                "created_at": datetime.datetime.now().isoformat()
            }
        elif endpoint == "/api/devices" and method == "GET":
            status = api_call["parameters"].get("status")
            devices = [
                {"id": "dev1", "name": "Device 1", "type": "switch", "status": "active"},
                {"id": "dev2", "name": "Device 2", "type": "router", "status": "active"},
                {"id": "dev3", "name": "Device 3", "type": "switch", "status": "inactive"}
            ]
            if status:
                devices = [dev for dev in devices if dev["status"] == status]
            return {"devices": devices}
        elif endpoint == "/api/simulations" and method == "POST":
            return {
                "simulation_id": "sim_" + str(uuid.uuid4())[:8],
                "network_id": api_call["parameters"].get("network_id", "unknown"),
                "status": "running",
                "estimated_completion": (datetime.datetime.now() + datetime.timedelta(seconds=api_call["parameters"].get("duration", 3600))).isoformat(),
                "created_at": datetime.datetime.now().isoformat()
            }
        else:
            return {
                "message": "Endpoint not implemented in simulation"
            }
    
    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get the history of processed requests
        
        Returns:
            List of processed requests
        """
        return self.request_history
    
    def clear_request_history(self):
        """Clear the request history"""
        self.request_history = []
        logger.info("Request history cleared")

# Usage:
# gateway = IntentGateway()
# response = await gateway.process_intent("Show me all active devices in the network")