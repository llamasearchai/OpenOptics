from typing import Dict, Any, List # For type hints
import logging
import random

logger = logging.getLogger(__name__)

class NetworkCostOptimizer:
    """Optimizes network designs for total cost of ownership (TCO)"""
    
    def __init__(self, component_library: Any, pricing_database: Any = None):
        """
        Initialize the NetworkCostOptimizer.
        Args:
            component_library: An instance of ComponentLibrary to fetch component data (including cost).
            pricing_database: Optional; can be used for more dynamic or bulk pricing if available.
                              If None, costs are assumed to be in component_library.
        """
        self.component_library = component_library
        self.pricing_database = pricing_database # Currently unused in this simplified version
        logger.info("NetworkCostOptimizer initialized.")
        
    def calculate_capex(self, network_design: Dict[str, Any]) -> float:
        """Calculate Capital Expenditure (CAPEX) based on components in the design."""
        design_id = network_design.get('id', 'N/A')
        logger.info(f"Calculating CAPEX for network design: {design_id}")
        total_capex = 0.0

        # Expecting network_design to have a list of component instances or references
        # Example structure: network_design["components"] = [{"id": "comp_id1", "quantity": 2}, ...]
        # Or network_design["nodes"] where each node has a component_id

        components_to_cost = []
        if "components" in network_design:
            for item in network_design["components"]:
                comp_id = item.get("id")
                quantity = item.get("quantity", 1)
                if comp_id:
                    components_to_cost.append({"id": comp_id, "quantity": quantity})
        elif "nodes" in network_design: # Alternative structure
            for node in network_design.get("nodes", []):
                comp_id = node.get("component_id") 
                quantity = node.get("quantity", 1) # Assuming nodes can represent multiple identical units
                if comp_id:
                    components_to_cost.append({"id": comp_id, "quantity": quantity})
        # Add similar logic for links if they have distinct costs beyond their constituent components (e.g., fiber spools listed as components)
        
        for item in components_to_cost:
            comp_data = self.component_library.get_component(item["id"])
            if comp_data:
                cost = comp_data.get("cost", 0.0) or 0.0 # Ensure None is treated as 0
                total_capex += cost * item["quantity"]
            else:
                logger.warning(f"CAPEX Calc: Component ID {item['id']} not found in library. Cost not added.")
        
        # Placeholder for installation costs, software licenses, etc.
        installation_factor = network_design.get("installation_factor", 0.1) # 10% of component cost
        total_capex += total_capex * installation_factor
        
        logger.info(f"Calculated CAPEX for {design_id}: ${total_capex:.2f}")
        return total_capex

    def calculate_opex(self, network_design: Dict[str, Any], annual_power_cost_per_kwh: float = 0.12) -> float:
        """Calculate annual Operational Expenditure (OPEX)."""
        design_id = network_design.get('id', 'N/A')
        logger.info(f"Calculating OPEX for network design: {design_id}")
        total_annual_opex = 0.0
        total_power_consumption_watts = 0.0

        components_for_opex = []
        if "components" in network_design:
            for item in network_design["components"]:
                comp_id = item.get("id")
                quantity = item.get("quantity", 1)
                if comp_id:
                    comp_data = self.component_library.get_component(comp_id)
                    if comp_data:
                        power = comp_data.get("power_consumption", 0.0) or 0.0 # Watts
                        total_power_consumption_watts += power * quantity
        elif "nodes" in network_design: # Alternative structure
             for node in network_design.get("nodes", []):
                comp_id = node.get("component_id") 
                quantity = node.get("quantity", 1)
                if comp_id:
                    comp_data = self.component_library.get_component(comp_id)
                    if comp_data:
                        power = comp_data.get("power_consumption", 0.0) or 0.0
                        total_power_consumption_watts += power * quantity

        # Annual power cost
        # Watts * hours_per_year / 1000 (to kWh) * cost_per_kWh
        hours_per_year = 24 * 365
        total_kwh_per_year = (total_power_consumption_watts * hours_per_year) / 1000.0
        annual_power_cost = total_kwh_per_year * annual_power_cost_per_kwh
        total_annual_opex += annual_power_cost

        # Placeholder for maintenance (e.g., 5% of CAPEX annually)
        # To avoid re-calculating CAPEX here, this ideally should be passed or estimated differently
        # For simplicity, let's assume a fixed maintenance cost or a factor of component count
        num_active_components = sum(item.get("quantity", 1) for item in network_design.get("components", [])) or \
                                sum(node.get("quantity",1) for node in network_design.get("nodes",[]))
        annual_maintenance_cost = num_active_components * 50 # $50 per active component placeholder
        total_annual_opex += annual_maintenance_cost
        
        # Placeholder for software subscriptions, site rental etc.
        total_annual_opex += network_design.get("annual_fixed_opex", 1000.0) # e.g. site costs

        logger.info(f"Calculated annual OPEX for {design_id}: ${total_annual_opex:.2f} (Power: ${annual_power_cost:.2f}, Maintenance: ${annual_maintenance_cost:.2f})")
        return total_annual_opex

    def generate_design_variants(self, base_design: Dict[str, Any], num_variants: int) -> List[Dict[str, Any]]:
        """Generate design variants by altering components or counts (simplified)."""
        logger.info(f"Generating {num_variants} design variants for: {base_design.get('id', 'BaseDesign')}")
        variants = []
        if not self.component_library:
            logger.error("Component library not available for generating variants.")
            return [base_design.copy() for _ in range(num_variants)] # Return copies if no library

        all_components_from_lib = self.component_library.get_all_components()
        if not all_components_from_lib:
            logger.warning("Component library is empty. Cannot generate meaningful variants.")
            return [base_design.copy() for _ in range(num_variants)]

        # Identify component types present in the base design to guide variation
        # This is a very naive variant generator. A real one would use heuristics, GA, etc.
        for i in range(num_variants):
            variant = base_design.copy() # Start with a copy
            variant["id"] = f"{base_design.get('id', 'design')}_v{i}"
            variant["components"] = list(base_design.get("components", [])) # Ensure it's a list for modification
            
            if not variant["components"]: # If no components, try to add one
                if all_components_from_lib:
                    chosen_comp = random.choice(all_components_from_lib)
                    variant["components"].append({"id": chosen_comp["id"], "quantity": 1})
            else:
                # Strategy: 50% chance to try swapping a component, 50% to change quantity
                if random.random() < 0.5 and variant["components"]: # Swap a component
                    idx_to_swap = random.randrange(len(variant["components"]))
                    original_item = variant["components"][idx_to_swap]
                    original_comp_data = self.component_library.get_component(original_item["id"])
                    
                    if original_comp_data:
                        # Find alternatives of the same type
                        alternatives = [c for c in all_components_from_lib if c["component_type"] == original_comp_data["component_type"] and c["id"] != original_item["id"]]
                        if alternatives:
                            new_comp = random.choice(alternatives)
                            variant["components"][idx_to_swap] = {"id": new_comp["id"], "quantity": original_item["quantity"]}
                            logger.info(f"Variant {i}: Swapped {original_item['id']} with {new_comp['id']}")
                elif variant["components"]: # Change quantity
                    idx_to_change_qty = random.randrange(len(variant["components"]))
                    original_qty = variant["components"][idx_to_change_qty]["quantity"]
                    # Change quantity by +/- 1 (ensure it's at least 1)
                    change = random.choice([-1,1])
                    new_qty = max(1, original_qty + change)
                    variant["components"][idx_to_change_qty]["quantity"] = new_qty
                    logger.info(f"Variant {i}: Changed quantity of {variant['components'][idx_to_change_qty]['id']} from {original_qty} to {new_qty}")
            variants.append(variant)
        return variants

    def evaluate_variants(self, variants: List[Dict[str, Any]], constraints: Dict[str, Any], forecast_years: int) -> Dict[str, Any]:
        """Placeholder for evaluating variants and selecting the best one based on TCO and constraints."""
        logger.info(f"Evaluating {len(variants)} variants with constraints: {constraints} for {forecast_years} years.")
        best_variant = None
        min_tco = float('inf')

        for variant in variants:
            # Here, one would check if 'variant' meets 'constraints'
            # For simplicity, we assume all variants meet constraints for this placeholder
            capex = self.calculate_capex(variant)
            opex = self.calculate_opex(variant)
            tco = capex + (opex * forecast_years)

            if tco < min_tco:
                min_tco = tco
                best_variant = variant
        
        if best_variant is None and variants: # Fallback if no variant was better (e.g. all had inf TCO)
            best_variant = variants[0]
            min_tco = self.calculate_capex(best_variant) + (self.calculate_opex(best_variant) * forecast_years)

        return {
            "design": best_variant, # This is the chosen design variant
            "capex": self.calculate_capex(best_variant) if best_variant else 0,
            "opex": self.calculate_opex(best_variant) if best_variant else 0,
            "tco": min_tco if best_variant else float('inf')
        }

    async def optimize_for_cost(self, network_design: Dict[str, Any], constraints: Dict[str, Any], forecast_years: int = 5):
        """Optimize network design for TCO while meeting performance constraints"""
        initial_capex = self.calculate_capex(network_design)
        annual_opex = self.calculate_opex(network_design)
        
        # TCO optimization with genetic algorithm
        variants = self.generate_design_variants(network_design, 100)
        optimized_design = self.evaluate_variants(variants, constraints, forecast_years)
        
        return {
            "original_tco": initial_capex + (annual_opex * forecast_years),
            "optimized_tco": optimized_design["capex"] + (optimized_design["opex"] * forecast_years),
            "savings_percentage": ((initial_capex + (annual_opex * forecast_years)) - 
                                  (optimized_design["capex"] + (optimized_design["opex"] * forecast_years))) / 
                                  (initial_capex + (annual_opex * forecast_years)) * 100,
            "optimized_design": optimized_design["design"]
        }