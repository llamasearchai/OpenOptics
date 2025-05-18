import datetime
import json
import copy
from typing import Dict, Any, List, Optional
import logging
import hashlib # For hashing in cache_key
import uuid # For ConfigRepository stub

logger = logging.getLogger(__name__)

# --- Placeholder/Stub Definitions for Dependencies ---

class BaseVendorAdapter:
    def __init__(self, vendor_name: str):
        self.vendor_name = vendor_name
        logger.info(f"{self.vendor_name}Adapter (stub) initialized.")

    async def connect(self):
        logger.info(f"{self.vendor_name}Adapter (stub): Connecting to mock devices.")
        # Mock connection logic

    async def apply_config(self, device_id: str, vendor_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.vendor_name}Adapter (stub): Applying config to device '{device_id}': {str(vendor_config)[:100]}...")
        return {"status": "success", "applied_config": vendor_config, "log": "Configuration applied successfully via mock adapter."}

    async def validate_config(self, device_id: str, vendor_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.vendor_name}Adapter (stub): Validating config for device '{device_id}': {str(vendor_config)[:100]}...")
        # Mock validation, always valid
        return {"valid": True, "errors": [], "warnings": []}

    async def get_device_config(self, device_id: str) -> Dict[str, Any]:
        logger.info(f"{self.vendor_name}Adapter (stub): Getting config for device '{device_id}'.")
        return {"hostname": device_id, "interfaces": {"GigabitEthernet0/0": {"ip_address": "192.168.1.1"}}, "vendor_specific_cli": f"show run | vendor {self.vendor_name}"}

class CiscoAdapter(BaseVendorAdapter):
    def __init__(self):
        super().__init__("Cisco")

class JuniperAdapter(BaseVendorAdapter):
    def __init__(self):
        super().__init__("Juniper")

class ConfigRepository:
    def __init__(self):
        logger.info("ConfigRepository (stub) initialized.")
        self._templates = {
            "vlan_config": {
                "description": "Standard VLAN configuration",
                "vendor_implementations": {
                    "Cisco": {"config_structure": {"vlan_id": None, "name": None}, "placeholders": {"vlan_id": "{{vlan_id}}","name": "{{vlan_name}}"}},
                    "Juniper": {"config_structure": {"vlans": { "default": {"vlan-id": None, "description": None}}}, "placeholders": {"vlans.default.vlan-id": "{{vlan_id}}", "vlans.default.description": "{{vlan_name}}"}}
                }
            }
        }
        self._devices = {
            "device_cisco_1": {"id": "device_cisco_1", "vendor": "Cisco", "model": "CSR1000v"},
            "device_juniper_1": {"id": "device_juniper_1", "vendor": "Juniper", "model": "vMX"}
        }
        self._device_configs: Dict[str, List[Dict[str, Any]]] = {}

    async def get_all_templates(self) -> Dict[str, Any]:
        logger.info("ConfigRepository (stub): Getting all templates.")
        return copy.deepcopy(self._templates) # Return a copy

    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"ConfigRepository (stub): Getting device '{device_id}'.")
        return self._devices.get(device_id)

    async def store_device_config(self, device_id: str, vendor_config: Dict[str, Any], 
                                  intent: Optional[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        config_id = str(uuid.uuid4())
        logger.info(f"ConfigRepository (stub): Storing config for device '{device_id}', ID {config_id}. Metadata: {metadata}")
        if device_id not in self._device_configs:
            self._device_configs[device_id] = []
        self._device_configs[device_id].append({
            "id": config_id, "vendor_config": vendor_config, "intent": intent, "metadata": metadata
        })
        return config_id

    async def update_template(self, config_type: str, template_data: Dict[str, Any]):
        logger.info(f"ConfigRepository (stub): Updating template for type '{config_type}'.")
        self._templates[config_type] = template_data

    async def get_all_devices(self) -> List[Dict[str, Any]]:
        logger.info("ConfigRepository (stub): Getting all devices.")
        return list(self._devices.values())

class AITranslator:
    def __init__(self):
        logger.info("AITranslator (stub) initialized.")

    async def translate_config(self, config_intent: Dict[str, Any], vendor: str) -> Dict[str, Any]:
        intent_type = config_intent.get("type", "unknown_intent")
        logger.info(f"AITranslator (stub): Translating intent '{intent_type}' to vendor '{vendor}'.")
        # Mock translation based on vendor - very simplified
        if vendor == "Cisco":
            return {"cisco_specific_key": f"value_for_{intent_type}", "parameters": config_intent.get("parameters")}
        elif vendor == "Juniper":
            return {"juniper_set_command": f"set system services {intent_type} value {config_intent.get('parameters')}"}
        return {"generic_config": f"config_for_{intent_type}", "vendor": vendor}

# --- End of Placeholder Definitions ---

class MultiVendorConfigManager:
    """Manages configurations across different network equipment vendors"""
    
    def __init__(self, vendor_adapters: Dict[str, BaseVendorAdapter], 
                 config_repository: ConfigRepository, 
                 ai_translator: AITranslator):
        self.vendor_adapters = vendor_adapters
        self.config_repository = config_repository
        self.ai_translator = ai_translator
        self.translation_cache = {}
        self.config_templates = {}
        
    async def initialize(self):
        """Initialize the multi-vendor configuration manager"""
        # Load vendor adapters
        for vendor, adapter in self.vendor_adapters.items():
            await adapter.connect()
            
        # Load config templates
        self.config_templates = await self.config_repository.get_all_templates()
        
        return {
            "status": "initialized",
            "vendors_supported": list(self.vendor_adapters.keys()),
            "templates_loaded": len(self.config_templates)
        }
    
    async def apply_unified_config(self, config_intent, target_devices):
        """Apply a vendor-neutral configuration intent to multiple vendor devices"""
        results = {}
        
        for device_id in target_devices:
            # Get device details
            device = await self.config_repository.get_device(device_id)
            vendor = device["vendor"]
            
            if vendor not in self.vendor_adapters:
                results[device_id] = {
                    "status": "error",
                    "message": f"Unsupported vendor: {vendor}"
                }
                continue
            
            # Translate vendor-neutral config to vendor-specific config
            vendor_config = await self._translate_to_vendor_specific(config_intent, vendor)
            
            # Apply configuration
            adapter = self.vendor_adapters[vendor]
            try:
                apply_result = await adapter.apply_config(device_id, vendor_config)
                results[device_id] = {
                    "status": "success",
                    "details": apply_result
                }
                
                # Store configuration in repository
                await self.config_repository.store_device_config(
                    device_id, 
                    vendor_config,
                    config_intent,
                    metadata={
                        "applied_by": "multi_vendor_manager",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                results[device_id] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return {
            "success_count": len([r for r in results.values() if r["status"] == "success"]),
            "error_count": len([r for r in results.values() if r["status"] == "error"]),
            "device_results": results
        }
    
    async def _translate_to_vendor_specific(self, config_intent, vendor):
        """Translate vendor-neutral config intent to vendor-specific configuration"""
        # Check cache first
        # Ensure config_intent is sortable for consistent hash if dicts are not ordered by default
        # For complex dicts, a more robust canonical serialization might be needed for caching key
        try:
            intent_json_str = json.dumps(config_intent, sort_keys=True)
        except TypeError:
            intent_json_str = str(config_intent) # Fallback if not JSON serializable
        
        cache_key = f"{vendor}:{hashlib.sha256(intent_json_str.encode()).hexdigest()}" # Use sha256 for better hash
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # First try template-based translation
        if config_intent["type"] in self.config_templates:
            template = self.config_templates[config_intent["type"]]
            if vendor in template["vendor_implementations"]:
                # Template available for this vendor
                vendor_template = template["vendor_implementations"][vendor]
                vendor_config = self._apply_template(vendor_template, config_intent["parameters"])
                self.translation_cache[cache_key] = vendor_config
                return vendor_config
        
        # Fall back to AI-based translation if template not available
        vendor_config = await self.ai_translator.translate_config(config_intent, vendor)
        
        # Cache the result
        self.translation_cache[cache_key] = vendor_config
        
        # Learn from this translation by storing as a new template
        await self._learn_new_template(config_intent, vendor, vendor_config)
        
        return vendor_config
    
    def _apply_template(self, template, parameters):
        """Apply parameters to a configuration template"""
        vendor_config = copy.deepcopy(template["config_structure"])
        
        # Replace placeholders with actual values
        for path, placeholder in template["placeholders"].items():
            parameter_name = placeholder.replace("{{", "").replace("}}", "")
            if parameter_name in parameters:
                value = parameters[parameter_name]
                self._set_nested_value(vendor_config, path.split('.'), value)
        
        return vendor_config
    
    def _set_nested_value(self, config, path, value):
        """Set a value in a nested dictionary using a path"""
        current = config
        for i, key in enumerate(path):
            if i == len(path) - 1:
                current[key] = value
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
    
    async def _learn_new_template(self, config_intent, vendor, vendor_config):
        """Learn from successful translations to improve future translations"""
        # Only store if this is a new configuration type or vendor
        config_type = config_intent["type"]
        
        if config_type not in self.config_templates:
            # Create new template entry
            self.config_templates[config_type] = {
                "description": f"Auto-generated template for {config_type}",
                "vendor_implementations": {
                    vendor: {
                        "config_structure": vendor_config,
                        "placeholders": self._detect_placeholders(config_intent, vendor_config)
                    }
                }
            }
        elif vendor not in self.config_templates[config_type]["vendor_implementations"]:
            # Add new vendor implementation to existing template
            self.config_templates[config_type]["vendor_implementations"][vendor] = {
                "config_structure": vendor_config,
                "placeholders": self._detect_placeholders(config_intent, vendor_config)
            }
        
        # Store updated templates
        await self.config_repository.update_template(config_type, self.config_templates[config_type])
    
    def _detect_placeholders(self, config_intent, vendor_config):
        """Automatically detect which parts of the config correspond to parameters"""
        placeholders = {}
        
        # Extract parameter values from the intent
        parameters = config_intent["parameters"]
        
        # Find these values in the vendor config
        for param_name, param_value in parameters.items():
            paths = self._find_value_paths(vendor_config, param_value)
            for path in paths:
                path_str = '.'.join(path)
                placeholders[path_str] = f"{{{{{param_name}}}}}"
        
        return placeholders
    
    def _find_value_paths(self, nested_dict, value, path=None):
        """Find all paths in a nested dictionary that contain a specific value"""
        if path is None:
            path = []
            
        paths = []
        
        if isinstance(nested_dict, dict):
            for k, v in nested_dict.items():
                if v == value:
                    paths.append(path + [k])
                elif isinstance(v, (dict, list)):
                    paths.extend(self._find_value_paths(v, value, path + [k]))
        elif isinstance(nested_dict, list):
            for i, item in enumerate(nested_dict):
                if item == value:
                    paths.append(path + [str(i)])
                elif isinstance(item, (dict, list)):
                    paths.extend(self._find_value_paths(item, value, path + [str(i)]))
        
        return paths
    
    async def validate_config(self, config_intent, target_devices):
        """Validate a configuration against target devices before applying"""
        validation_results = {}
        
        for device_id in target_devices:
            # Get device details
            device = await self.config_repository.get_device(device_id)
            vendor = device["vendor"]
            
            if vendor not in self.vendor_adapters:
                validation_results[device_id] = {
                    "status": "error",
                    "valid": False,
                    "message": f"Unsupported vendor: {vendor}"
                }
                continue
            
            # Translate to vendor-specific config
            vendor_config = await self._translate_to_vendor_specific(config_intent, vendor)
            
            # Validate configuration
            adapter = self.vendor_adapters[vendor]
            try:
                validation = await adapter.validate_config(device_id, vendor_config)
                validation_results[device_id] = {
                    "status": "success",
                    "valid": validation["valid"],
                    "warnings": validation.get("warnings", []),
                    "errors": validation.get("errors", [])
                }
            except Exception as e:
                validation_results[device_id] = {
                    "status": "error",
                    "valid": False,
                    "message": str(e)
                }
        
        return {
            "overall_valid": all(r.get("valid", False) for r in validation_results.values()),
            "device_results": validation_results
        }
    
    async def backup_device_configs(self, device_ids=None):
        """Backup configurations from network devices"""
        if device_ids is None:
            # Get all managed devices
            devices = await self.config_repository.get_all_devices()
            device_ids = [d["id"] for d in devices]
        
        backup_results = {}
        
        for device_id in device_ids:
            # Get device details
            device = await self.config_repository.get_device(device_id)
            vendor = device["vendor"]
            
            if vendor not in self.vendor_adapters:
                backup_results[device_id] = {
                    "status": "error",
                    "message": f"Unsupported vendor: {vendor}"
                }
                continue
            
            # Backup configuration
            adapter = self.vendor_adapters[vendor]
            try:
                config = await adapter.get_device_config(device_id)
                
                # Store the backup
                backup_id = await self.config_repository.store_device_config(
                    device_id,
                    config,
                    None,  # No intent for backups
                    metadata={
                        "type": "backup",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                
                backup_results[device_id] = {
                    "status": "success",
                    "backup_id": backup_id
                }
            except Exception as e:
                backup_results[device_id] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return {
            "success_count": len([r for r in backup_results.values() if r["status"] == "success"]),
            "error_count": len([r for r in backup_results.values() if r["status"] == "error"]),
            "device_results": backup_results
        }