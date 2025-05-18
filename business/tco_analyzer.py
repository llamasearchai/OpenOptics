import datetime
import math
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# --- Placeholder/Stub Definitions for Dependencies ---

class InventoryService:
    def __init__(self):
        logger.info("InventoryService (stub) initialized.")
        self._inventory_data = {
            "components": {
                "transceivers": [{"model": "TRX100", "quantity": 100, "id": "t1"}],
                "switches": [{"model": "SW-CORE", "quantity": 2, "id": "s1"}]
            },
            "software": [{"name": "NOS-BASIC", "tier": "standard", "id": "sw1"}],
            "stats": {
                "total_ports": 200,
                "total_capacity_gbps": 10000,
                "rack_units": 40,
                "device_count": 102,
                "total_weight_kg": 500
            }
        }

    async def get_current_inventory(self) -> Dict[str, Any]:
        logger.info("InventoryService (stub): Getting current inventory.")
        return self._inventory_data

    async def project_inventory(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"InventoryService (stub): Projecting inventory with changes: {changes}")
        # Simplified: return current inventory as a base for projection
        # A real implementation would apply 'changes' to create a new inventory state.
        projected_inventory = self._inventory_data.copy() # Shallow copy, careful with nested dicts if modifying
        projected_inventory["stats"] = self._inventory_data["stats"].copy()
        if changes.get("add_switches"): # Example very basic change application
            projected_inventory["stats"]["device_count"] += changes["add_switches"]
        return projected_inventory

class PowerMonitor:
    def __init__(self):
        logger.info("PowerMonitor (stub) initialized.")

    async def get_projected_power_consumption(self, inventory: Dict[str, Any]) -> Dict[str, float]:
        num_devices = inventory.get("stats", {}).get("device_count", 10)
        logger.info(f"PowerMonitor (stub): Projecting power for {num_devices} devices.")
        # Dummy calculation: 100W per device, 24/7 for a year
        annual_kwh = num_devices * 100 * 24 * 365 / 1000
        return {"annual_kwh": annual_kwh, "peak_watts": num_devices * 150}

class MaintenanceTracker:
    def __init__(self):
        logger.info("MaintenanceTracker (stub) initialized.")

    async def get_annual_maintenance_cost(self, inventory: Dict[str, Any]) -> float:
        num_devices = inventory.get("stats", {}).get("device_count", 10)
        cost_per_device = 50 # Dummy cost
        logger.info(f"MaintenanceTracker (stub): Calculating annual maintenance for {num_devices} devices.")
        return float(num_devices * cost_per_device)

class CostDatabase:
    def __init__(self):
        logger.info("CostDatabase (stub) initialized.")
        self._component_costs = {"TRX100": 500.0, "SW-CORE": 10000.0, "default": 100.0}
        self._software_costs = {("NOS-BASIC", "standard"): 2000.0, ("default", "default"): 500.0}
        self._sw_maintenance_costs = {("NOS-BASIC", "standard"): 200.0, ("default", "default"): 50.0}
        self._power_cost_kwh = 0.12
        self._rack_unit_cost_annual = 100.0
        self._support_cost_device_annual = 20.0
        self._disposal_cost_kg = 0.5
        self._labor_rate_hour = 75.0

    async def get_component_cost(self, model_name: str) -> float:
        cost = self._component_costs.get(model_name, self._component_costs["default"])
        logger.info(f"CostDatabase (stub): Cost for component '{model_name}': {cost}")
        return cost

    async def get_software_cost(self, sw_name: str, sw_tier: str) -> float:
        cost = self._software_costs.get((sw_name, sw_tier), self._software_costs[("default", "default")])
        logger.info(f"CostDatabase (stub): Cost for software '{sw_name} ({sw_tier})': {cost}")
        return cost

    async def get_power_cost_per_kwh(self) -> float:
        logger.info(f"CostDatabase (stub): Power cost/kWh: {self._power_cost_kwh}")
        return self._power_cost_kwh

    async def get_rack_unit_cost(self) -> float: # Assuming annual
        logger.info(f"CostDatabase (stub): Rack unit annual cost: {self._rack_unit_cost_annual}")
        return self._rack_unit_cost_annual

    async def get_support_cost_per_device(self) -> float: # Assuming annual
        logger.info(f"CostDatabase (stub): Support cost/device annual: {self._support_cost_device_annual}")
        return self._support_cost_device_annual

    async def get_software_maintenance_cost(self, sw_name: str, sw_tier: str) -> float: # Assuming annual
        cost = self._sw_maintenance_costs.get((sw_name, sw_tier), self._sw_maintenance_costs[("default", "default")])
        logger.info(f"CostDatabase (stub): Software maintenance cost for '{sw_name} ({sw_tier})': {cost}")
        return cost

    async def get_disposal_cost_per_kg(self) -> float:
        logger.info(f"CostDatabase (stub): Disposal cost/kg: {self._disposal_cost_kg}")
        return self._disposal_cost_kg

    async def get_labor_rate(self) -> float: # Assuming per hour
        logger.info(f"CostDatabase (stub): Labor rate/hour: {self._labor_rate_hour}")
        return self._labor_rate_hour

# --- End of Placeholder Definitions ---

class NetworkTCOAnalyzer:
    """Advanced Total Cost of Ownership analysis for optical networks"""
    
    def __init__(self, inventory_service: InventoryService, power_monitor: PowerMonitor, 
                 maintenance_tracker: MaintenanceTracker, cost_database: CostDatabase):
        self.inventory_service = inventory_service
        self.power_monitor = power_monitor
        self.maintenance_tracker = maintenance_tracker
        self.cost_database = cost_database
        self.analysis_history = []
    
    async def calculate_current_tco(self, timeframe_years=5):
        """Calculate the current Total Cost of Ownership for the network"""
        # Get current inventory
        inventory = await self.inventory_service.get_current_inventory()
        
        # Calculate capital expenditures
        capex = await self._calculate_capex(inventory)
        
        # Calculate operational expenditures
        opex = await self._calculate_opex(inventory, timeframe_years)
        
        # Calculate refresh costs
        refresh_costs = await self._calculate_refresh_costs(inventory, timeframe_years)
        
        # Calculate end-of-life costs
        eol_costs = await self._calculate_eol_costs(inventory, timeframe_years)
        
        # Calculate TCO
        total_tco = capex + opex + refresh_costs + eol_costs
        
        # Prepare analysis result
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "timeframe_years": timeframe_years,
            "total_tco": total_tco,
            "capex": capex,
            "opex": opex,
            "refresh_costs": refresh_costs,
            "eol_costs": eol_costs,
            "annual_breakdown": await self._calculate_annual_breakdown(inventory, timeframe_years),
            "per_category_breakdown": await self._calculate_category_breakdown(inventory, timeframe_years),
            "kpis": {
                "tco_per_port": total_tco / inventory["stats"]["total_ports"],
                "tco_per_gbps": total_tco / inventory["stats"]["total_capacity_gbps"],
                "opex_to_capex_ratio": opex / capex if capex > 0 else float('inf'),
                "annual_tco": total_tco / timeframe_years
            }
        }
        
        # Store analysis in history
        self.analysis_history.append(analysis)
        
        return analysis
    
    async def compare_deployment_scenarios(self, scenarios):
        """Compare TCO for different deployment scenarios"""
        results = {}
        
        # Calculate TCO for each scenario
        for scenario_id, scenario in scenarios.items():
            # Get scenario inventory
            projected_inventory = await self.inventory_service.project_inventory(scenario["changes"])
            
            # Calculate TCO components
            capex = await self._calculate_capex(projected_inventory, scenario.get("capex_adjustments", {}))
            opex = await self._calculate_opex(projected_inventory, scenario["timeframe_years"], 
                                          scenario.get("opex_adjustments", {}))
            refresh_costs = await self._calculate_refresh_costs(projected_inventory, scenario["timeframe_years"],
                                                            scenario.get("refresh_adjustments", {}))
            eol_costs = await self._calculate_eol_costs(projected_inventory, scenario["timeframe_years"])
            
            # Calculate TCO
            total_tco = capex + opex + refresh_costs + eol_costs
            
            results[scenario_id] = {
                "name": scenario["name"],
                "description": scenario["description"],
                "timeframe_years": scenario["timeframe_years"],
                "total_tco": total_tco,
                "capex": capex,
                "opex": opex,
                "refresh_costs": refresh_costs,
                "eol_costs": eol_costs,
                "annual_breakdown": await self._calculate_annual_breakdown(projected_inventory, scenario["timeframe_years"]),
                "kpis": {
                    "tco_per_port": total_tco / projected_inventory["stats"]["total_ports"],
                    "tco_per_gbps": total_tco / projected_inventory["stats"]["total_capacity_gbps"],
                    "opex_to_capex_ratio": opex / capex if capex > 0 else float('inf'),
                    "annual_tco": total_tco / scenario["timeframe_years"],
                    "capex_percentage": (capex / total_tco) * 100,
                    "opex_percentage": (opex / total_tco) * 100,
                    "refresh_percentage": (refresh_costs / total_tco) * 100,
                    "eol_percentage": (eol_costs / total_tco) * 100
                }
            }
        
        # Identify optimal scenario based on total TCO
        optimal_scenario = min(results.items(), key=lambda x: x[1]["total_tco"])[0]
        
        # Calculate comparisons between scenarios
        comparisons = {}
        baseline_scenario = scenarios.get("baseline", list(scenarios.keys())[0])
        baseline_tco = results[baseline_scenario]["total_tco"]
        
        for scenario_id, scenario_result in results.items():
            if scenario_id != baseline_scenario:
                tco_difference = scenario_result["total_tco"] - baseline_tco
                percentage_difference = (tco_difference / baseline_tco) * 100
                
                comparisons[scenario_id] = {
                    "baseline": baseline_scenario,
                    "tco_difference": tco_difference,
                    "percentage_difference": percentage_difference,
                    "is_cheaper": tco_difference < 0,
                    "annual_savings": -tco_difference / scenario_result["timeframe_years"] if tco_difference < 0 else 0
                }
        
        return {
            "scenarios": results,
            "optimal_scenario": optimal_scenario,
            "comparisons": comparisons
        }
    
    async def _calculate_capex(self, inventory, adjustments=None):
        """Calculate capital expenditures for network components"""
        adjustments = adjustments or {}
        total_capex = 0
        
        # Hardware costs
        for category, items in inventory["components"].items():
            category_total = 0
            for item in items:
                item_cost = await self.cost_database.get_component_cost(item["model"])
                quantity = item["quantity"]
                category_total += item_cost * quantity
            
            # Apply category adjustments if any
            adjustment_factor = adjustments.get(category, {}).get("factor", 1.0)
            total_capex += category_total * adjustment_factor
        
        # Installation costs
        installation_cost = total_capex * 0.15  # Typically 15% of hardware costs
        installation_adjustment = adjustments.get("installation", {}).get("factor", 1.0)
        total_capex += installation_cost * installation_adjustment
        
        # Initial software/licensing costs
        software_costs = 0
        for sw_item in inventory.get("software", []):
            sw_cost = await self.cost_database.get_software_cost(sw_item["name"], sw_item["tier"])
            software_costs += sw_cost
        
        software_adjustment = adjustments.get("software", {}).get("factor", 1.0)
        total_capex += software_costs * software_adjustment
        
        return total_capex
    
    async def _calculate_opex(self, inventory, timeframe_years, adjustments=None):
        """Calculate operational expenditures over the specified timeframe"""
        adjustments = adjustments or {}
        annual_opex = 0
        
        # Power costs
        power_consumption = await self.power_monitor.get_projected_power_consumption(inventory)
        annual_power_cost = power_consumption["annual_kwh"] * await self.cost_database.get_power_cost_per_kwh()
        power_adjustment = adjustments.get("power", {}).get("factor", 1.0)
        annual_opex += annual_power_cost * power_adjustment
        
        # Cooling costs (typically 30% of power costs)
        annual_cooling_cost = annual_power_cost * 0.3
        cooling_adjustment = adjustments.get("cooling", {}).get("factor", 1.0)
        annual_opex += annual_cooling_cost * cooling_adjustment
        
        # Maintenance costs
        maintenance_cost = await self.maintenance_tracker.get_annual_maintenance_cost(inventory)
        maintenance_adjustment = adjustments.get("maintenance", {}).get("factor", 1.0)
        annual_opex += maintenance_cost * maintenance_adjustment
        
        # Space/rack costs
        space_cost = inventory["stats"]["rack_units"] * await self.cost_database.get_rack_unit_cost()
        space_adjustment = adjustments.get("space", {}).get("factor", 1.0)
        annual_opex += space_cost * space_adjustment
        
        # Support costs (staff, etc.)
        support_cost = inventory["stats"]["device_count"] * await self.cost_database.get_support_cost_per_device()
        support_adjustment = adjustments.get("support", {}).get("factor", 1.0)
        annual_opex += support_cost * support_adjustment
        
        # Software subscription/maintenance
        sw_maintenance_cost = 0
        for sw_item in inventory.get("software", []):
            annual_sw_cost = await self.cost_database.get_software_maintenance_cost(sw_item["name"], sw_item["tier"])
            sw_maintenance_cost += annual_sw_cost
        
        sw_adjustment = adjustments.get("software_maintenance", {}).get("factor", 1.0)
        annual_opex += sw_maintenance_cost * sw_adjustment
        
        # Total OPEX over timeframe
        total_opex = annual_opex * timeframe_years
        
        # Account for cost escalation over time (typically 3% annually)
        escalation_rate = adjustments.get("escalation_rate", 0.03)
        if escalation_rate > 0:
            escalation_factor = 0
            for year in range(timeframe_years):
                escalation_factor += (1 + escalation_rate) ** year
            
            total_opex = annual_opex * escalation_factor
        
        return total_opex
    
    async def _calculate_refresh_costs(self, inventory, timeframe_years, adjustments=None):
        """Calculate technology refresh costs over the timeframe"""
        adjustments = adjustments or {}
        total_refresh_cost = 0
        
        # Get typical refresh cycles for different equipment types
        refresh_cycles = {
            "transceivers": 5,
            "switches": 5,
            "routers": 7,
            "amplifiers": 8,
            "monitoring_equipment": 6
        }
        
        # Apply adjustments to refresh cycles if provided
        for category, cycle in refresh_cycles.items():
            adjustment = adjustments.get(f"{category}_cycle", 1.0)
            refresh_cycles[category] = math.ceil(cycle * adjustment)
        
        # Calculate refresh costs by category
        for category, items in inventory["components"].items():
            if category not in refresh_cycles:
                continue
                
            cycle = refresh_cycles[category]
            refreshes_in_timeframe = max(0, timeframe_years - cycle) // cycle
            
            if refreshes_in_timeframe > 0:
                category_value = 0
                for item in items:
                    item_cost = await self.cost_database.get_component_cost(item["model"])
                    quantity = item["quantity"]
                    category_value += item_cost * quantity
                
                # Apply depreciation factor
                depreciation_factor = adjustments.get(f"{category}_depreciation", 0.7)
                refresh_cost = category_value * depreciation_factor * refreshes_in_timeframe
                
                total_refresh_cost += refresh_cost
        
        return total_refresh_cost
    
    async def _calculate_eol_costs(self, inventory, timeframe_years):
        """Calculate end-of-life disposal and decommissioning costs"""
        total_eol_cost = 0
        
        # Only include EOL costs if timeframe exceeds average component lifecycle
        avg_lifecycle = 7  # Average lifecycle in years
        
        if timeframe_years >= avg_lifecycle:
            # Calculate disposal costs based on weight
            total_weight_kg = inventory["stats"].get("total_weight_kg", 0)
            disposal_cost_per_kg = await self.cost_database.get_disposal_cost_per_kg()
            
            # Decommissioning labor costs
            device_count = inventory["stats"]["device_count"]
            decommission_hours_per_device = 2  # Average hours to decommission a device
            labor_rate = await self.cost_database.get_labor_rate()
            
            decommission_cost = device_count * decommission_hours_per_device * labor_rate
            disposal_cost = total_weight_kg * disposal_cost_per_kg
            
            total_eol_cost = decommission_cost + disposal_cost
        
        return total_eol_cost
    
    async def _calculate_annual_breakdown(self, inventory, timeframe_years):
        """Calculate cost breakdown by year"""
        annual_costs = []
        
        # Get base annual costs
        base_annual_opex = await self._calculate_opex(inventory, 1)  # Just one year
        
        for year in range(1, timeframe_years + 1):
            year_costs = {
                "year": year,
                "opex": base_annual_opex * (1.03 ** (year - 1)),  # Apply 3% annual increase
                "capex": 0,
                "refresh": 0,
                "total": 0
            }
            
            # Add refresh costs for applicable years
            refresh_cycles = {
                "transceivers": 5,
                "switches": 5,
                "routers": 7,
                "amplifiers": 8,
                "monitoring_equipment": 6
            }
            
            for category, cycle in refresh_cycles.items():
                if year > cycle and year % cycle == 0:
                    category_value = 0
                    for item in inventory["components"].get(category, []):
                        item_cost = await self.cost_database.get_component_cost(item["model"])
                        quantity = item["quantity"]
                        category_value += item_cost * quantity
                    
                    # Apply 70% cost for refresh (typical discount from new)
                    year_costs["refresh"] += category_value * 0.7
            
            # Add CAPEX only for year 1
            if year == 1:
                year_costs["capex"] = await self._calculate_capex(inventory)
            
            # Calculate EOL costs for final year
            if year == timeframe_years:
                year_costs["eol"] = await self._calculate_eol_costs(inventory, timeframe_years)
            else:
                year_costs["eol"] = 0
            
            # Calculate total for the year
            year_costs["total"] = sum(v for k, v in year_costs.items() if k != "year")
            
            annual_costs.append(year_costs)
        
        return annual_costs
    
    async def _calculate_category_breakdown(self, inventory, timeframe_years):
        """Calculate TCO breakdown by component category"""
        category_costs = {}
        
        # Calculate costs by component category
        for category, items in inventory["components"].items():
            category_value = 0
            for item in items:
                item_cost = await self.cost_database.get_component_cost(item["model"])
                quantity = item["quantity"]
                category_value += item_cost * quantity
            
            # Calculate maintenance over timeframe (typically 10% of value annually)
            maintenance_cost = category_value * 0.1 * timeframe_years
            
            # Calculate refresh costs
            refresh_cycles = {
                "transceivers": 5,
                "switches": 5,
                "routers": 7,
                "amplifiers": 8,
                "monitoring_equipment": 6
            }
            
            refresh_cost = 0
            if category in refresh_cycles:
                cycle = refresh_cycles[category]
                refreshes_in_timeframe = max(0, timeframe_years - cycle) // cycle
                if refreshes_in_timeframe > 0:
                    refresh_cost = category_value * 0.7 * refreshes_in_timeframe
            
            category_costs[category] = {
                "capex": category_value,
                "maintenance": maintenance_cost,
                "refresh": refresh_cost,
                "total": category_value + maintenance_cost + refresh_cost
            }
        
        return category_costs