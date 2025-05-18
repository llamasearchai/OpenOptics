#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenOptics: AI-Enhanced Optical Network Architecture Framework

An advanced system combining traditional optical networking expertise with cutting-edge
AI capabilities for designing, evaluating, optimizing, and testing optical networking
infrastructure for hyperscale AI computing environments.

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
import datetime
import uuid
import math
import random
import threading
import multiprocessing
import queue
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Type, Callable, Generator
from dataclasses import dataclass, field, asdict
from functools import lru_cache
import asyncio
import copy

# Data analysis and scientific computing
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
import torch.nn as nn

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# Network simulation and modeling
import simpy
import networkx as nx

# Hardware and system monitoring
import psutil
import gputil
import prometheus_client as prom

# Web and API
from flask import Flask
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import fastapi
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn

# Database
import sqlite3
import sqlalchemy
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import sqlite_utils # Added for ComponentLibrary

# LangChain integration
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, initialize_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler

# OpenAI API integration
import openai
from openai import OpenAI as DirectOpenAI

# Neo4j for knowledge graph
from neo4j import GraphDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openoptics.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SIMULATION_DIR = BASE_DIR / "simulations"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MODEL_DIR = BASE_DIR / "models"

# Create directories
for directory in [CONFIG_DIR, DATA_DIR, RESULTS_DIR, SIMULATION_DIR, KNOWLEDGE_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

CONFIG_PATH = CONFIG_DIR / "openoptics_config.yaml"

# Load configuration
def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Create default config
        default_config = {
            "component_library": {
                "database_path": "data/components.db",
                "auto_load_common": True
            },
            "evaluation": {
                "power_budget_margin": 3.0,
                "reliability_threshold": 0.8,
                "technology_weights": {
                    "power_efficiency": 0.25,
                    "density": 0.20,
                    "cost": 0.20,
                    "maturity": 0.15,
                    "reach": 0.10,
                    "reliability": 0.10
                }
            },
            "simulation": {
                "default_duration": 24.0,
                "packet_size_distribution": {
                    "64": 0.4,
                    "512": 0.3,
                    "1500": 0.3
                },
                "default_traffic_pattern": "diurnal",
                "default_traffic_intensity": 0.7,
                "default_failure_rate": 0.1,
                "default_failure_duration": 1.0,
                "result_dir": "simulations"
            },
            "testing": {
                "standard_test": {
                    "duration": 24.0,
                    "temperature": 25.0,
                    "humidity": 50.0
                },
                "thermal_cycling_test": {
                    "duration": 48.0,
                    "temp_min": 0.0,
                    "temp_max": 70.0,
                    "cycle_count": 10
                },
                "stress_test": {
                    "duration": 24.0,
                    "temperature": 35.0,
                    "intensity": 0.9
                },
                "result_dir": "results/testing"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "enable_docs": True,
                "cors": {
                    "allowed_origins": ["*"],
                    "allowed_methods": ["*"],
                    "allowed_headers": ["*"]
                }
            },
            "system": {
                "log_level": "INFO",
                "seed": 42,
                "parallel_workers": 4
            },
            "ai": {
                "openai_api_key": "",
                "model": "gpt-4",
                "embedding_model": "text-embedding-ada-002",
                "temperature": 0.7,
                "max_tokens": 2000,
                "knowledge_update_interval": 24  # hours
            },
            "knowledge_graph": {
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_user": "neo4j",
                "neo4j_password": "password"
            },
            "agents": {
                "enable_optimization_agent": True,
                "enable_design_assistant_agent": True,
                "enable_failure_analysis_agent": True,
                "enable_network_monitoring_agent": True,
                "agent_check_interval": 60  # seconds
            }
        }
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(default_config, f, sort_keys=False)
        return default_config

CONFIG = load_config()

#############################################################
# Placeholder Core Component Definitions
#############################################################

class ComponentLibrary:
    """Placeholder for the Optical Component Library."""
    DB_PATH = CONFIG_DIR.parent / DATA_DIR / "components.db" # Correct path construction

    def __init__(self):
        logger.info(f"ComponentLibrary initialized. Database path: {self.DB_PATH}")
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure data directory exists
        self.db = sqlite_utils.Database(self.DB_PATH) # Use sqlite-utils
        self._create_table_if_not_exists()
        if self._is_empty():
            self._populate_initial_data()

    def _create_table_if_not_exists(self):
        components_table = self.db["components"]
        if not components_table.exists():
            logger.info(f"Creating 'components' table in {self.DB_PATH}")
            components_table.create({
                "id": str, # Primary Key
                "name": str,
                "component_type": str, # e.g., 'transceiver', 'switch', 'fiber'
                "manufacturer": str,
                "model_number": str,
                "data_rate_gbps": int,
                "reach_km": float,
                "wavelength_nm": float,
                "power_consumption_w": float,
                "cost_usd": float,
                "dimensions_mm": str, # e.g., "72x18x8.5"
                "weight_g": float,
                "operating_temp_c": str, # e.g., "0-70"
                "mtbf_hours": int,
                "technology": str, # e.g., "SiPh", "InP", "VCSEL"
                "interface_type": str, # e.g., "QSFP28", "LC", "MPO"
                "compliance_standards": str, # e.g., "IEEE 802.3ae", "OIF"
                "features": str, # JSON string for additional features like DDM, CDR
                "additional_specs": str, # JSON string for type-specific specs
                # Transceiver specific (can be NULL for other types)
                "form_factor": str,
                "tx_power_dbm": float,
                "rx_sensitivity_dbm": float,
                "supported_modulations": str, # JSON list: e.g., ["NRZ", "PAM4"]
                # Switch specific
                "port_count": int,
                "switching_capacity_tbps": float,
                "latency_ns": float,
                # Fiber specific
                "fiber_type": str, # e.g., "SMF", "MMF"
                "core_diameter_um": float,
                "attenuation_db_per_km": float,
                "dispersion_ps_per_nm_km": float
            }, pk="id")
            logger.info("'components' table created successfully or already exists.")
        else:
            logger.info("'components' table already exists.")

    def _is_empty(self) -> bool:
        return self.db["components"].count == 0

    def _populate_initial_data(self):
        logger.info(f"Populating 'components' table with initial data in {self.DB_PATH}...")
        initial_components_data = [
            {
                "id": "TRX001", "name": "QSFP28-LR4", "component_type": "transceiver", 
                "manufacturer": "OptiCore", "model_number": "OC-Q28-LR4-100G", 
                "data_rate_gbps": 100, "reach_km": 10, "wavelength_nm": 1310, 
                "power_consumption_w": 3.5, "cost_usd": 150.0, "form_factor": "QSFP28",
                "tx_power_dbm": 2.0, "rx_sensitivity_dbm": -10.5,
                "features": json.dumps({"DDM": True, "CDR": True}), 
                "supported_modulations": json.dumps(["NRZ"]),
                "technology": "InP", "interface_type": "LC",
                "additional_specs": json.dumps({"operating_case_temp_c": "0-70"})
            },
            {
                "id": "TRX002", "name": "QSFP-DD-DR4", "component_type": "transceiver", 
                "manufacturer": "PhotonWorks", "model_number": "PW-QDD-DR4-400G", 
                "data_rate_gbps": 400, "reach_km": 0.5, "wavelength_nm": 1310, 
                "power_consumption_w": 12.0, "cost_usd": 600.0, "form_factor": "QSFP-DD",
                "tx_power_dbm": 1.0, "rx_sensitivity_dbm": -5.0,
                "features": json.dumps({"DDM": True, "CMIS": True}), 
                "supported_modulations": json.dumps(["PAM4"]),
                "technology": "SiPh", "interface_type": "MPO-12",
                "additional_specs": json.dumps({"lanes": 4, "modulation_per_lane": "100G PAM4"})
            },
            {
                "id": "SW001", "name": "EdgeRouter XG", "component_type": "switch",
                "manufacturer": "NetFabric", "model_number": "NFX-ER-XG-64",
                "port_count": 64, "switching_capacity_tbps": 12.8, "latency_ns": 300,
                "power_consumption_w": 800, "cost_usd": 15000, 
                "features": json.dumps({"programmable_fabric": True, "telemetry_streaming": True}),
                "data_rate_gbps": 100 # Port speed
            },
             {
                "id": "FIB001", "name": "SMF-28 Ultra", "component_type": "fiber",
                "manufacturer": "Corning", "model_number": "SMF-28U",
                "fiber_type": "SMF", "core_diameter_um": 9.0,
                "attenuation_db_per_km": 0.18, "dispersion_ps_per_nm_km": 17.0,
                "cost_usd": 0.5, # Per meter
                "features": json.dumps({"low_water_peak": True})
            }
        ]
        try:
            self.db["components"].insert_all(initial_components_data, pk="id", replace=True)
            logger.info(f"Successfully populated 'components' table with {len(initial_components_data)} items.")
        except Exception as e:
            logger.error(f"Error populating 'components' table: {e}")

    def add_component(self, component_data: Dict[str, Any]) -> bool:
        if "id" not in component_data:
            logger.error("Component data must include an 'id'.")
            return False
        try:
            # Ensure all fields defined in the schema are present, fill with None if missing
            # This is important for sqlite-utils to not complain about missing columns
            # if the table schema is strict (which it is by default on insert)
            table_columns = self.db["components"].columns_dict.keys()
            for col in table_columns:
                if col not in component_data:
                    component_data[col] = None
            
            # Convert JSON fields to strings if they are dicts/lists
            for key, value in component_data.items():
                if isinstance(value, (dict, list)):
                    component_data[key] = json.dumps(value)
            
            self.db["components"].upsert(component_data, pk="id", alter=True) # alter=True allows adding new columns if any
            logger.info(f"Component '{component_data['id']}' added/updated in the library.")
            return True
        except Exception as e:
            logger.error(f"Error adding/updating component '{component_data.get('id', 'N/A')}': {e}")
            return False

    def find_transceivers(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Basic implementation. Complex queries on JSON fields might need raw SQL or specific indexing.
        # sqlite-utils .rows_where is good for simple key-value matches.
        base_query = "SELECT * FROM components WHERE component_type = :component_type"
        params = {"component_type": "transceiver"}
        
        conditions = []
        for key, value in criteria.items():
            if key == "component_type": continue # Already handled
            if key in ["data_rate_gbps", "reach_km", "cost_usd", "tx_power_dbm", "rx_sensitivity_dbm"]: # Numeric fields
                # For numeric fields, allow for ranges if value is a tuple (min, max) or comparison (>, <, >=, <=)
                if isinstance(value, tuple) and len(value) == 2:
                    conditions.append(f"{key} >= :{key}_min AND {key} <= :{key}_max")
                    params[f"{key}_min"] = value[0]
                    params[f"{key}_max"] = value[1]
                elif isinstance(value, str) and value.startswith((">=", "<=", ">", "<")):
                    op = ">=" if value.startswith(">=") else "<=" if value.startswith("<=") else ">" if value.startswith(">") else "<"
                    num_val = float(value[len(op):])
                    conditions.append(f"{key} {op} :{key}_val")
                    params[f"{key}_val"] = num_val
                else: # Exact match
                    conditions.append(f"{key} = :{key}")
                    params[key] = value
            elif key in ["manufacturer", "form_factor", "technology", "interface_type", "name"]: # Text fields (exact match or LIKE)
                if "%" in str(value): # Assume LIKE for strings with wildcards
                    conditions.append(f"{key} LIKE :{key}")
                else: # Exact match
                    conditions.append(f"{key} = :{key}")
                params[key] = value
            elif key == "supported_modulations": # Example of JSON query (basic LIKE)
                 conditions.append(f"json_extract(supported_modulations, '$[?(@ == \\\"{value}\\\")]') IS NOT NULL")
                 # Or more simply, if it's a list of strings:
                 # conditions.append(f"supported_modulations LIKE :supported_modulations_like")
                 # params["supported_modulations_like"] = f"%\\\"{value}\\\"%")


        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        logger.info(f"Finding transceivers with query: {base_query} and params: {params}")
        try:
            results = list(self.db.query(base_query, params))
            # Post-process JSON strings back to Python objects if needed by consumer
            for row in results:
                if row.get("features") and isinstance(row["features"], str):
                    try: row["features"] = json.loads(row["features"]) 
                    except json.JSONDecodeError: pass
                if row.get("supported_modulations") and isinstance(row["supported_modulations"], str):
                    try: row["supported_modulations"] = json.loads(row["supported_modulations"])
                    except json.JSONDecodeError: pass
                if row.get("additional_specs") and isinstance(row["additional_specs"], str):
                    try: row["additional_specs"] = json.loads(row["additional_specs"])
                    except json.JSONDecodeError: pass
            return results
        except Exception as e:
            logger.error(f"Error finding transceivers: {e}")
            return []

    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        try:
            component = self.db["components"].get(component_id)
            if component:
                # Post-process JSON strings
                if component.get("features") and isinstance(component["features"], str):
                    try: component["features"] = json.loads(component["features"])
                    except json.JSONDecodeError: pass
                if component.get("supported_modulations") and isinstance(component["supported_modulations"], str):
                    try: component["supported_modulations"] = json.loads(component["supported_modulations"])
                    except json.JSONDecodeError: pass
                if component.get("additional_specs") and isinstance(component["additional_specs"], str):
                    try: component["additional_specs"] = json.loads(component["additional_specs"])
                    except json.JSONDecodeError: pass
                return dict(component) # Ensure it's a plain dict
            return None
        except Exception as e:
            logger.error(f"Error getting component {component_id}: {e}")
            return None

    def get_all_components(self) -> List[Dict[str, Any]]:
        try:
            components = list(self.db["components"].rows)
             # Post-process JSON strings
            for component in components:
                if component.get("features") and isinstance(component["features"], str):
                    try: component["features"] = json.loads(component["features"])
                    except json.JSONDecodeError: pass
                if component.get("supported_modulations") and isinstance(component["supported_modulations"], str):
                    try: component["supported_modulations"] = json.loads(component["supported_modulations"])
                    except json.JSONDecodeError: pass
                if component.get("additional_specs") and isinstance(component["additional_specs"], str):
                    try: component["additional_specs"] = json.loads(component["additional_specs"])
                    except json.JSONDecodeError: pass
            return [dict(c) for c in components] # Ensure plain dicts
        except Exception as e:
            logger.error(f"Error getting all components: {e}")
            return []

class OpticalEvaluator:
    """Placeholder for the Optical Network Evaluator."""
    def __init__(self, component_library: ComponentLibrary):
        self.component_library = component_library
        logger.info("OpticalEvaluator initialized.")

    def evaluate_network_design(self, topology: Any, components: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"OpticalEvaluator: Evaluating network design with {len(components)} component types and {len(links)} links.")
        
        total_cost = 0.0
        total_power_consumption = 0.0
        warnings = []
        recommendations = []
        evaluation_score = 100.0 # Start with a perfect score

        if not components:
            warnings.append("No components provided for evaluation.")
            evaluation_score -= 20

        # Assume components list might contain dicts with 'id' and 'quantity', or just component data
        # For simplicity, let's assume `components` is a list of component data already fetched, each appearing once.
        # A real design would have quantities or instances of components.
        # Let's refine this to expect a list of component IDs used in the design.
        # The actual component data (cost, power) will be fetched from the library.

        actual_components_in_design = [] # List of component dicts from library
        if isinstance(topology, dict) and "nodes" in topology:
            for node in topology.get("nodes", []):
                comp_id = node.get("component_id")
                if comp_id:
                    comp_data = self.component_library.get_component(comp_id)
                    if comp_data:
                        actual_components_in_design.append(comp_data)
                        total_cost += comp_data.get("cost", 0.0) or 0.0
                        total_power_consumption += comp_data.get("power_consumption", 0.0) or 0.0
                    else:
                        warnings.append(f"Component ID {comp_id} in topology not found in library.")
                        evaluation_score -= 5
                else:
                    warnings.append(f"Node {node.get('id')} in topology missing component_id.")
                    evaluation_score -= 2
        else:
             warnings.append("Topology data is missing or not in expected format (nodes list).")
             evaluation_score -=10

        # Simple link validation (example: check if links connect known component types)
        if links:
            for link_idx, link in enumerate(links):
                source_id = link.get("source") # Expecting component ID
                target_id = link.get("target") # Expecting component ID
                # Further checks could involve link capacity vs component data rate, fiber type vs transceiver type, etc.
                if not source_id or not target_id:
                    warnings.append(f"Link {link_idx} is missing source or target.")
                    evaluation_score -= 2
        else:
            warnings.append("No links provided in the design for evaluation.")
            # This might be fine for a single-component test, but not for a network.

        power_budget_margin = CONFIG.get("evaluation", {}).get("power_budget_margin", 3.0)
        # Placeholder for power budget calculation per link
        # For each link: TxPower - RxSensitivity - LinkLoss (fiber_attenuation * length + connector_loss) > margin
        # This requires more detailed link and component data (e.g., fiber lengths, connector types)

        if total_power_consumption > 10000: # Arbitrary threshold
            warnings.append(f"High total power consumption: {total_power_consumption:.2f}W")
            recommendations.append("Review component selection for more power-efficient alternatives.")
            evaluation_score -= total_power_consumption / 1000 # Penalty based on excess

        if total_cost > 100000: # Arbitrary threshold
            warnings.append(f"High total estimated cost: ${total_cost:.2f}")
            recommendations.append("Explore cost optimization strategies or alternative components.")
            evaluation_score -= total_cost / 10000 # Penalty

        # Normalize score to be between 0 and 1.0, or 0-100
        evaluation_score = max(0, min(100, evaluation_score))
        status = "success" if not warnings else ("warning" if evaluation_score > 50 else "error")

        return {
            "status": status,
            "evaluation_score": round(evaluation_score / 100.0, 2), # Score from 0.0 to 1.0
            "power_consumption_watts": round(total_power_consumption, 2),
            "estimated_cost": round(total_cost, 2),
            "bottlenecks": [], # Requires more sophisticated analysis
            "warnings": warnings,
            "recommendations": recommendations
        }

    def optimize_component_selection(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"OpticalEvaluator: Optimizing component selection for requirements: {requirements}")
        # Placeholder - e.g., find best transceivers for a given reach/capacity
        if requirements.get("type") == "transceiver":
            return self.component_library.find_transceivers(requirements)
        return [{"optimized_component": "placeholder_optimized_comp", "reason": "met requirements"}]

    def compare_optical_technologies(self, technologies: List[str]) -> Dict[str, Any]:
        logger.info(f"OpticalEvaluator: Comparing technologies: {technologies}")
        # Placeholder comparison
        comparison = {}
        for tech in technologies:
            comparison[tech] = {"pros": ["pro1", "pro2"], "cons": ["con1"], "suitability_score": random.uniform(0.5, 0.9)}
        return comparison

class NetworkSimulator:
    """Simulator for optical network performance and failures."""
    
    def __init__(self, component_library=None, evaluator=None):
        self.component_library = component_library
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__ + ".NetworkSimulator")
        self.logger.info("NetworkSimulator initialized")
        
    def simulate_network(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate network performance based on configuration."""
        try:
            self.logger.info(f"Simulating network with config: {config.get('name', 'unnamed')}")
            
            # Extract configuration parameters
            topology = config.get("topology", {})
            components = config.get("components", [])
            links = config.get("links", [])
            duration_hours = config.get("duration_hours", 24)
            load_factor = config.get("load_factor", 0.7)  # 70% load by default
            
            # If components are specified by ID, look them up
            if self.component_library and all(isinstance(c, str) for c in components):
                components = [self.component_library.get_component(c_id) for c_id in components]
                components = [c for c in components if c is not None]
            
            # Build the network model
            network_model = self._build_network_model(topology, components, links)
            
            # Run the simulation
            performance_metrics = self._simulate_performance(network_model, duration_hours, load_factor)
            
            # Simulate events
            events = self._simulate_events(network_model, duration_hours)
            
            # Combine results
            results = {
                "simulation_id": str(uuid.uuid4()),
                "config_name": config.get("name", "unnamed"),
                "duration_hours": duration_hours,
                "performance_metrics": performance_metrics,
                "events": events,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error simulating network: {e}")
            return {"error": str(e)}
            
    def _build_network_model(self, topology, components, links):
        """Build a network model for simulation."""
        # Create a network graph
        graph = {"nodes": [], "edges": [], "components": {}}
        
        # Process nodes
        for node_id, node_data in topology.get("nodes", {}).items():
            graph["nodes"].append({
                "id": node_id,
                "type": node_data.get("type", "generic"),
                "location": node_data.get("location", {}),
                "components": []
            })
        
        # Add components to nodes
        for component in components:
            comp_id = component.get("id")
            node_id = component.get("node_id")
            
            if comp_id and node_id:
                # Find the node
                node = next((n for n in graph["nodes"] if n["id"] == node_id), None)
                
                if node:
                    node["components"].append(comp_id)
                    graph["components"][comp_id] = component
        
        # Process links
        for link_id, link_data in topology.get("links", {}).items():
            source = link_data.get("source")
            target = link_data.get("target")
            
            if source and target:
                graph["edges"].append({
                    "id": link_id,
                    "source": source,
                    "target": target,
                    "distance_km": link_data.get("distance_km", 0),
                    "fiber_type": link_data.get("fiber_type", "smf"),
                    "capacity_gbps": link_data.get("capacity_gbps", 0),
                    "channels": link_data.get("channels", [])
                })
        
        # If links are provided as a list instead of a dict
        if isinstance(links, list):
            for link_data in links:
                source = link_data.get("source")
                target = link_data.get("target")
                
                if source and target:
                    link_id = link_data.get("id", f"{source}-{target}")
                    graph["edges"].append({
                        "id": link_id,
                        "source": source,
                        "target": target,
                        "distance_km": link_data.get("distance_km", 0),
                        "fiber_type": link_data.get("fiber_type", "smf"),
                        "capacity_gbps": link_data.get("capacity_gbps", 0),
                        "channels": link_data.get("channels", [])
                    })
        
        return graph
            
    def _simulate_performance(self, network_model, duration_hours, load_factor):
        """Simulate network performance over time."""
        # Initialize performance metrics
        metrics = {
            "throughput_gbps": [],
            "latency_ms": [],
            "packet_loss_percent": [],
            "osnr_db": [],
            "power_consumption_w": [],
            "timestamps": []
        }
        
        # Calculate base values
        base_throughput = self._calculate_base_throughput(network_model)
        base_latency = self._calculate_base_latency(network_model)
        base_osnr = self._calculate_base_osnr(network_model)
        base_power = self._calculate_power_consumption(network_model)
        
        # Calculate time steps
        time_step_hours = 1.0  # 1-hour increments
        steps = int(duration_hours / time_step_hours)
        
        # Generate time-series data
        for step in range(steps):
            # Calculate timestamp
            timestamp = datetime.datetime.now() + datetime.timedelta(hours=step * time_step_hours)
            
            # Calculate time-varying load factor based on time of day
            hour_of_day = timestamp.hour
            time_factor = self._get_time_of_day_factor(hour_of_day)
            current_load = load_factor * time_factor
            
            # Simulate some inefficiency, throughput is not always 100% of load if network is stressed
            efficiency = max(0.8, 1.0 - (current_load - 0.7) * 0.5) if current_load > 0.7 else 1.0
            
            # Calculate metrics with variations
            throughput = base_throughput * current_load * efficiency
            
            # Latency increases with load
            latency_factor = 1.0 + max(0, (current_load - 0.7) * 0.5)
            latency = base_latency * latency_factor
            
            # Packet loss increases exponentially with high load
            packet_loss = 0.001 if current_load < 0.8 else 0.001 * math.exp((current_load - 0.8) * 10)
            
            # OSNR degrades slightly with load
            osnr = base_osnr - (current_load * 2)
            
            # Power consumption increases with load
            power = base_power * (0.7 + 0.3 * current_load)
            
            # Add random variations (1-2%)
            throughput *= random.uniform(0.98, 1.02)
            latency *= random.uniform(0.98, 1.02)
            packet_loss *= random.uniform(0.95, 1.05)
            osnr *= random.uniform(0.99, 1.01)
            power *= random.uniform(0.99, 1.01)
            
            # Store metrics
            metrics["throughput_gbps"].append(throughput)
            metrics["latency_ms"].append(latency)
            metrics["packet_loss_percent"].append(packet_loss * 100)  # Convert to percentage
            metrics["osnr_db"].append(osnr)
            metrics["power_consumption_w"].append(power)
            metrics["timestamps"].append(timestamp.isoformat())
        
        # Calculate summary statistics
        summary = {
            "avg_throughput_gbps": sum(metrics["throughput_gbps"]) / len(metrics["throughput_gbps"]),
            "max_throughput_gbps": max(metrics["throughput_gbps"]),
            "min_throughput_gbps": min(metrics["throughput_gbps"]),
            "avg_latency_ms": sum(metrics["latency_ms"]) / len(metrics["latency_ms"]),
            "max_latency_ms": max(metrics["latency_ms"]),
            "avg_packet_loss_percent": sum(metrics["packet_loss_percent"]) / len(metrics["packet_loss_percent"]),
            "avg_osnr_db": sum(metrics["osnr_db"]) / len(metrics["osnr_db"]),
            "min_osnr_db": min(metrics["osnr_db"]),
            "avg_power_consumption_w": sum(metrics["power_consumption_w"]) / len(metrics["power_consumption_w"])
        }
        
        return {
            "time_series": metrics,
            "summary": summary
        }
    
    def _simulate_events(self, network_model, duration_hours):
        """Simulate network events such as failures and alarms."""
        # Calculate the number of events based on duration and reliability
        # Higher reliability = fewer events
        
        component_count = len(network_model.get("components", {}))
        link_count = len(network_model.get("edges", []))
        
        # Base event rate: events per hour
        base_event_rate = 0.01  # 1 event per 100 hours on average
        
        # Calculate expected number of events
        expected_events = base_event_rate * duration_hours * (component_count + link_count) / 100
        
        # Poisson distribution for number of events
        num_events = np.random.poisson(expected_events)
        
        # Generate events
        events_logged = []
        
        for i in range(num_events):
            # Random time within the simulation period
            event_time = datetime.datetime.now() + datetime.timedelta(
                hours=random.uniform(0, duration_hours)
            )
            
            # Determine if it's a component or link event
            is_component_event = random.random() < (component_count / (component_count + link_count))
            
            if is_component_event and component_count > 0:
                # Component event
                component_ids = list(network_model["components"].keys())
                component_id = random.choice(component_ids)
                component = network_model["components"][component_id]
                
                event = {
                    "type": "component_event",
                    "component_id": component_id,
                    "component_type": component.get("type", "unknown"),
                    "severity": random.choice(["info", "warning", "critical"]),
                    "timestamp": event_time.isoformat(),
                    "details": f"Simulated event {i+1} on component {component_id}."
                }
            elif link_count > 0:
                # Link event
                edges = network_model["edges"]
                link = random.choice(edges)
                
                event = {
                    "type": "link_event",
                    "link_id": link["id"],
                    "source": link["source"],
                    "target": link["target"],
                    "severity": random.choice(["info", "warning", "critical"]),
                    "timestamp": event_time.isoformat(),
                    "details": f"Simulated event {i+1} on a random element."
                }
            else:
                # Generic event if no components or links
                event = {
                    "type": "system_event",
                    "severity": random.choice(["info", "warning", "critical"]),
                    "timestamp": event_time.isoformat(),
                    "details": f"Simulated event {i+1} on the system."
                }
            
            events_logged.append(event)
        
        # Sort events by timestamp
        events_logged.sort(key=lambda e: e["timestamp"])
        
        return {
            "simulated_events_count": len(events_logged),
            "simulated_events": events_logged
        }
    
    def compare_topologies(self, topologies, config):
        """Compare multiple topologies using the same simulation parameters."""
        results = {}
        
        for i, topology in enumerate(topologies):
            # Create a config for this topology
            topo_config = copy.deepcopy(config)
            topo_config["topology"] = topology
            
            # Run simulation
            results[f"topology_{i}_{config.get('name', '')}"] = self.simulate_network(topo_config)
        
        return results
    
    def simulate_failure_scenarios(self, config: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate various failure scenarios and their impact on the network."""
        base_simulation = self.simulate_network(config)
        scenario_simulations = {}
        
        for i, scenario in enumerate(scenarios):
            # Create a modified config based on the scenario
            scenario_config = copy.deepcopy(config)
            
            # Apply scenario modifications
            self._apply_scenario_to_config(scenario_config, scenario)
            
            # Run simulation with the modified config
            scenario_sim = self.simulate_network(scenario_config)
            
            # Calculate impact
            impact = self._calculate_scenario_impact(base_simulation, scenario_sim)
            
            # Store results
            scenario_simulations[f"scenario_{i}_{scenario.get('name', '')}"] = {
                "scenario": scenario,
                "simulation": scenario_sim,
                "impact": impact
            }
        
        return {
            "base_simulation": base_simulation,
            "scenarios": scenario_simulations
        }
    
    def _apply_scenario_to_config(self, config, scenario):
        """Apply a failure scenario to a configuration."""
        scenario_type = scenario.get("type", "")
        
        if scenario_type == "component_failure":
            # Handle component failure
            component_id = scenario.get("component_id")
            
            if component_id:
                # Find and modify the component
                for comp in config.get("components", []):
                    if isinstance(comp, dict) and comp.get("id") == component_id:
                        comp["status"] = "failed"
                        break
        
        elif scenario_type == "link_failure":
            # Handle link failure
            link_id = scenario.get("link_id")
            
            if link_id:
                # Find and modify the link
                for link in config.get("links", []):
                    if isinstance(link, dict) and link.get("id") == link_id:
                        link["status"] = "failed"
                        break
        
        elif scenario_type == "partial_degradation":
            # Handle partial degradation
            component_id = scenario.get("component_id")
            degradation_factor = scenario.get("degradation_factor", 0.5)
            
            if component_id:
                # Find and modify the component
                for comp in config.get("components", []):
                    if isinstance(comp, dict) and comp.get("id") == component_id:
                        comp["degradation_factor"] = degradation_factor
                        break
        
        elif scenario_type == "traffic_surge":
            # Handle traffic surge
            surge_factor = scenario.get("surge_factor", 1.5)
            config["load_factor"] = config.get("load_factor", 0.7) * surge_factor
    
    def _calculate_scenario_impact(self, base_sim, scenario_sim):
        """Calculate the impact of a scenario compared to the base simulation."""
        impact = {}
        
        # Extract summary metrics
        base_summary = base_sim.get("performance_metrics", {}).get("summary", {})
        scenario_summary = scenario_sim.get("performance_metrics", {}).get("summary", {})
        
        # Calculate impact for each metric
        for metric in ["avg_throughput_gbps", "avg_latency_ms", "avg_packet_loss_percent", 
                      "avg_osnr_db", "avg_power_consumption_w"]:
            if metric in base_summary and metric in scenario_summary:
                base_value = base_summary[metric]
                scenario_value = scenario_summary[metric]
                
                if base_value != 0:
                    percent_change = (scenario_value - base_value) / base_value * 100
                else:
                    percent_change = float('inf') if scenario_value > 0 else 0
                
                impact[f"{metric}_change"] = percent_change
        
        # Calculate overall impact score
        throughput_impact = abs(impact.get("avg_throughput_gbps_change", 0))
        latency_impact = abs(impact.get("avg_latency_ms_change", 0))
        packet_loss_impact = abs(impact.get("avg_packet_loss_percent_change", 0))
        
        # Weighted impact score
        overall_impact = (
            0.5 * throughput_impact + 
            0.3 * latency_impact + 
            0.2 * packet_loss_impact
        ) / 100  # Normalize to 0-1 scale
        
        impact["overall_impact_score"] = min(1.0, overall_impact)
        
        return impact
    
    def _calculate_base_throughput(self, network_model):
        """Calculate the base throughput capacity of the network."""
        throughput = 0
        
        # Sum up the capacity of all edges
        for edge in network_model.get("edges", []):
            throughput += edge.get("capacity_gbps", 0)
        
        # If no capacity was specified on edges, estimate from components
        if throughput == 0:
            for comp_id, component in network_model.get("components", {}).items():
                if component.get("type") == "transceiver":
                    throughput += component.get("capacity_gbps", 0)
        
        return max(throughput, 100)  # Minimum of 100 Gbps for simulation
    
    def _calculate_base_latency(self, network_model):
        """Calculate the base latency of the network."""
        total_distance = 0
        
        # Sum up the distance of all edges
        for edge in network_model.get("edges", []):
            total_distance += edge.get("distance_km", 0)
        
        # Calculate latency based on distance
        # Speed of light in fiber: ~200,000 km/s (2/3 of c)
        # 1 ms per 200 km, plus 0.1 ms per hop for equipment latency
        hop_count = len(network_model.get("edges", []))
        
        latency = (total_distance / 200) + (hop_count * 0.1)
        
        return max(latency, 1.0)  # Minimum of 1 ms for simulation
    
    def _calculate_base_osnr(self, network_model):
        """Calculate the base OSNR of the network."""
        # Typical OSNR values for optical networks: 15-25 dB
        return 20.0  # Default OSNR
    
    def _calculate_power_consumption(self, network_model):
        """Calculate the power consumption of the network."""
        power = 0
        
        # Sum up power consumption of all components
        for comp_id, component in network_model.get("components", {}).items():
            power += component.get("power_consumption_w", 0)
        
        # If no power was specified, estimate based on component types
        if power == 0:
            for comp_id, component in network_model.get("components", {}).items():
                comp_type = component.get("type", "")
                
                if comp_type == "transceiver":
                    capacity = component.get("capacity_gbps", 100)
                    power += capacity / 10  # ~10 W per 100G
                elif comp_type == "amplifier":
                    power += 20  # ~20 W
                elif comp_type == "roadm":
                    degrees = component.get("degrees", 4)
                    power += degrees * 25  # ~25 W per degree
        
        return max(power, 100)  # Minimum of 100 W for simulation
    
    def _get_time_of_day_factor(self, hour):
        """Get a load factor based on time of day (0-23)."""
        # Define traffic patterns
        # - Night (0-5): Low traffic (40-60%)
        # - Morning (6-9): Increasing traffic (60-100%)
        # - Day (10-16): High traffic (90-100%)
        # - Evening (17-21): Decreasing traffic (70-90%)
        # - Night (22-23): Low traffic (50-70%)
        
        if 0 <= hour <= 5:
            return 0.4 + (hour / 5) * 0.2  # 40-60%
        elif 6 <= hour <= 9:
            return 0.6 + (hour - 6) / 3 * 0.4  # 60-100%
        elif 10 <= hour <= 16:
            return 0.9 + (hour - 10) / 6 * 0.1  # 90-100%
        elif 17 <= hour <= 21:
            return 0.9 - (hour - 17) / 4 * 0.2  # 90-70%
        else:  # 22-23
            return 0.7 - (hour - 22) / 1 * 0.2  # 70-50%

class OpticalTester:
    """Placeholder for the Optical Component/System Tester."""
    def __init__(self, component_library: ComponentLibrary):
        logger.info("OpticalTester initialized.")
        self.component_library = component_library

    def analyze_test_results(self, component_id: str, test_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"OpticalTester: Analyzing test results for component: {component_id}")
        
        component = self.component_library.get_component(component_id)
        if not component:
            logger.error(f"Component {component_id} not found in library for testing.")
            return {
                "component_id": component_id,
                "status": "error",
                "message": "Component not found in library."
            }

        test_conditions = test_conditions or {}
        test_duration_hours = test_conditions.get("duration_hours", 24)
        test_temperature_celsius = test_conditions.get("temperature_celsius", 25)
        stress_level_percent = test_conditions.get("stress_level_percent", 0) # 0-100

        # --- Simulate test measurements based on component type and conditions ---
        measurements = {}
        anomalies_detected = False
        test_status = "pass"
        issues_found = []

        base_reliability = 0.99 # Base probability of passing perfectly

        # Simulate temperature effects (very simplified)
        if test_temperature_celsius > 50 or test_temperature_celsius < 0:
            base_reliability -= 0.1
            issues_found.append(f"Extreme temperature ({test_temperature_celsius}°C) applied.")
        
        # Simulate stress effects
        if stress_level_percent > 75:
            base_reliability -= (stress_level_percent - 75) / 250.0 # up to -0.1 at 100% stress
            issues_found.append(f"High stress level ({stress_level_percent}%) applied.")

        if component.get("component_type") == "transceiver":
            # Nominal values from component data, if available
            nominal_tx_power = component.get("tx_power", 0.0) # dBm
            nominal_rx_sensitivity = component.get("rx_sensitivity", -10.0) # dBm
            
            # Simulate measured Tx Power: slightly varies around nominal
            tx_power_drift = random.uniform(-0.5, 0.5) * (1 + stress_level_percent / 100.0)
            measurements["tx_power_dbm"] = round(nominal_tx_power + tx_power_drift, 2)

            # Simulate measured Rx Sensitivity: can degrade (become less sensitive, i.e., higher value)
            rx_sensitivity_drift = random.uniform(-0.2, 1.0) * (1 + stress_level_percent / 100.0)
            measurements["rx_sensitivity_dbm"] = round(nominal_rx_sensitivity + rx_sensitivity_drift, 2)
            
            measurements["osnr_db"] = round(random.uniform(30, 40) - (stress_level_percent / 20.0), 1) # OSNR degrades with stress
            measurements["ber"] = random.uniform(1e-12, 1e-9) * (1 + stress_level_percent / 10.0) # BER increases

            if measurements["tx_power_dbm"] < nominal_tx_power - 1.0:
                issues_found.append(f"Tx power significantly lower than nominal.")
                base_reliability -= 0.05
            if measurements["rx_sensitivity_dbm"] > nominal_rx_sensitivity + 1.0:
                 issues_found.append(f"Rx sensitivity significantly worse than nominal.")
                 base_reliability -= 0.05
            if measurements["osnr_db"] < 32:
                issues_found.append(f"OSNR is marginal: {measurements['osnr_db']} dB.")
                base_reliability -= 0.02

        elif component.get("component_type") == "switch":
            measurements["port_error_rate_avg"] = random.uniform(0, 1e-7) * (1 + stress_level_percent / 50.0)
            measurements["forwarding_latency_us"] = random.uniform(1, 5) * (1 + stress_level_percent / 100.0)
            measurements["power_draw_watts"] = (component.get("power_consumption", 100)) * random.uniform(0.9, 1.2) 
            if measurements["port_error_rate_avg"] > 1e-8:
                issues_found.append(f"Average port error rate is high.")
                base_reliability -= 0.05
        
        elif component.get("component_type") == "fiber":
            nominal_attenuation = component.get("attenuation", 0.2) # dB/km
            fiber_length_km = float(component.get("reach", "1km").replace("km","")) if component.get("reach") else 1.0
            measurements["total_insertion_loss_db"] = round((nominal_attenuation + random.uniform(-0.02, 0.05)) * fiber_length_km, 2)
            measurements["reflectance_db"] = round(random.uniform(-50, -35) + (stress_level_percent/10.0),1) # Reflectance worsens
            if measurements["total_insertion_loss_db"] > (nominal_attenuation + 0.03) * fiber_length_km * 1.1: # 10% over expected max
                 issues_found.append(f"Insertion loss higher than expected.")
                 base_reliability -= 0.1
        else:
            measurements["generic_metric"] = random.uniform(0, 100)
            issues_found.append("No specific test protocol for this component type, generic check done.")
            base_reliability -=0.01

        # Determine final status
        if random.random() > base_reliability:
            test_status = "fail"
            anomalies_detected = True
            if not issues_found:
                 issues_found.append("General reliability failure detected during stress period.")
        elif issues_found:
            test_status = "warning"
            anomalies_detected = True # Warnings are also anomalies from ideal
        
        return {
            "component_id": component_id,
            "component_name": component.get("name"),
            "component_type": component.get("component_type"),
            "test_conditions": test_conditions,
            "test_date": datetime.datetime.now().isoformat(),
            "status": test_status,
            "measurements": measurements,
            "anomalies_detected": anomalies_detected,
            "issues_found": issues_found
        }

    def compare_components(self, component_ids: List[str], test_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"OpticalTester: Comparing components: {component_ids}")
        comparison = {}
        for cid in component_ids:
            comparison[cid] = self.analyze_test_results(cid, test_conditions) # Simplified
        return {"comparison_summary": "ComponentA performed better in metric X.", "details": comparison}


#############################################################
# Core Data Models and AI Enhancement 
#############################################################

# Keep original data models from the initial code...
# [OpticalTechnology, WavelengthTechnology, FormFactor, NetworkComponent, etc.]
# For now, we'll add very basic dataclass stubs for these to make AgentManager.update_knowledge work.

@dataclass
class BaseComponent:
    id: str
    name: str
    component_type: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    data_rate: Optional[str] = None # Example common field

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TransceiverComponent(BaseComponent):
    technology: Optional[str] = None
    wavelength_tech: Optional[str] = None
    form_factor: Optional[str] = None
    reach: Optional[str] = None # e.g. "10km"
    tx_power: Optional[float] = None # dBm
    rx_sensitivity: Optional[float] = None # dBm
    component_type: str = "transceiver" # Override default

# AI-Enhanced Models

@dataclass
class AIModelConfig:
    """Configuration for AI models"""
    model_name: str
    temperature: float
    max_tokens: int
    api_key: str
    embedding_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class AgentConfig:
    """Configuration for autonomous agents"""
    agent_id: str
    agent_type: str
    enabled: bool
    check_interval: int  # seconds
    tools: List[str]
    llm_config: AIModelConfig
    memory_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

class KnowledgeGraphManager:
    """Manages the optical networking knowledge graph"""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the knowledge graph manager
        
        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self.connect()
        
    def connect(self):
        """Connect to the Neo4j database"""
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Connected to Neo4j knowledge graph")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None
    
    def close(self):
        """Close the connection to Neo4j"""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def add_component(self, component: Any):
        """Add a component to the knowledge graph
        
        Args:
            component: Component to add
        """
        if not self._driver:
            logger.error("Not connected to Neo4j database")
            return
        
        # Extract component details
        props = component.to_dict()
        component_type = props.get("component_type", "Unknown")
        
        with self._driver.session() as session:
            # Create component node
            session.run(
                "MERGE (c:Component {id: $id}) "
                "SET c.name = $name, c.component_type = $component_type, "
                "c.manufacturer = $manufacturer, c.model = $model, "
                "c.updated_at = timestamp() "
                "RETURN c",
                id=props.get("id"),
                name=props.get("name"),
                component_type=component_type,
                manufacturer=props.get("manufacturer", ""),
                model=props.get("model", "")
            )
            
            # Add type-specific properties
            if hasattr(component, "technology"):
                session.run(
                    "MATCH (c:Component {id: $id}) "
                    "SET c.technology = $technology, c.wavelength_tech = $wavelength_tech, "
                    "c.form_factor = $form_factor, c.data_rate = $data_rate "
                    "RETURN c",
                    id=props.get("id"),
                    technology=props.get("technology", ""),
                    wavelength_tech=props.get("wavelength_tech", ""),
                    form_factor=props.get("form_factor", ""),
                    data_rate=props.get("data_rate", "")
                )
    
    def add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None):
        """Add a relationship between components
        
        Args:
            source_id: Source component ID
            target_id: Target component ID
            rel_type: Relationship type
            properties: Relationship properties
        """
        if not self._driver:
            logger.error("Not connected to Neo4j database")
            return
        
        if properties is None:
            properties = {}
        
        with self._driver.session() as session:
            session.run(
                f"MATCH (a:Component {{id: $source_id}}), (b:Component {{id: $target_id}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"SET r += $properties "
                f"RETURN a, r, b",
                source_id=source_id,
                target_id=target_id,
                properties=properties
            )
    
    def query(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query against the knowledge graph
        
        Args:
            cypher_query: Cypher query
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if not self._driver:
            logger.error("Not connected to Neo4j database")
            return []
        
        if parameters is None:
            parameters = {}
        
        with self._driver.session() as session:
            result = session.run(cypher_query, **parameters)
            return [record.data() for record in result]
    
    def get_component_knowledge(self, component_id: str) -> Dict[str, Any]:
        """Get knowledge about a component
        
        Args:
            component_id: Component ID
            
        Returns:
            Component knowledge
        """
        if not self._driver:
            logger.error("Not connected to Neo4j database")
            return {}
        
        with self._driver.session() as session:
            result = session.run(
                "MATCH (c:Component {id: $id}) "
                "OPTIONAL MATCH (c)-[r]->(related) "
                "RETURN c, collect(distinct {type: type(r), target: related}) as relationships",
                id=component_id
            )
            
            for record in result:
                component = record["c"]
                relationships = record["relationships"]
                
                # Compose a knowledge dictionary
                return {
                    "id": component["id"],
                    "name": component["name"],
                    "type": component["component_type"],
                    "properties": {k: v for k, v in component.items() if k not in ["id", "name", "component_type"]},
                    "relationships": relationships
                }
        
        return {}

class VectorKnowledgeBase:
    """Vector store for optical networking knowledge"""
    
    def __init__(self, embedding_model: str, persist_directory: str = None):
        """Initialize the vector knowledge base
        
        Args:
            embedding_model: OpenAI embedding model name
            persist_directory: Directory to persist vector store
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or str(KNOWLEDGE_DIR / "vector_store")
        
        # Initialize embedding function
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the vector store from disk"""
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embeddings
            )
            logger.info(f"Loaded vector store from {self.persist_directory}")
        except Exception as e:
            logger.info(f"Creating new vector store: {e}")
            # Create an empty vector store
            self.vectorstore = FAISS.from_texts(
                ["OpenOptics initialization"], 
                self.embeddings
            )
            # Save it
            self.vectorstore.save_local(self.persist_directory)
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add texts to the vector store
        
        Args:
            texts: List of texts to add
            metadatas: List of metadata for each text
        """
        if not texts:
            return
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        self.vectorstore.add_texts(texts, metadatas)
        self.vectorstore.save_local(self.persist_directory)
        logger.info(f"Added {len(texts)} texts to vector store")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store
        
        Args:
            documents: List of documents to add
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.add_texts(texts, metadatas)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vector store for similar documents
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get documents relevant to a query
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.similarity_search(query, k=k)

class OpticsTool:
    """Tool for AI agents to use in the optical networking domain"""
    
    def __init__(self, name: str, description: str, func: Callable, is_async: bool = False):
        """Initialize the optics tool
        
        Args:
            name: Tool name
            description: Tool description
            func: Tool function
            is_async: Whether the function is async
        """
        self.name = name
        self.description = description
        self.func = func
        self.is_async = is_async
    
    def __call__(self, *args, **kwargs):
        """Call the tool function"""
        return self.func(*args, **kwargs)
    
    def to_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool
        
        Returns:
            LangChain Tool
        """
        return Tool(
            name=self.name,
            description=self.description,
            func=self.func
        )

class OpticsAgent:
    """Base class for optical networking agents"""
    
    def __init__(self, config: AgentConfig, knowledge_base: VectorKnowledgeBase, tools: List[OpticsTool]):
        """Initialize the optics agent
        
        Args:
            config: Agent configuration
            knowledge_base: Vector knowledge base
            tools: List of available tools
        """
        self.config = config
        self.knowledge_base = knowledge_base
        self.tools = tools
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_config.model_name,
            temperature=config.llm_config.temperature,
            max_tokens=config.llm_config.max_tokens,
            api_key=config.llm_config.api_key
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=config.memory_size
        )
        
        # Initialize agent
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent"""
        # Convert tools to LangChain tools
        lc_tools = [tool.to_langchain_tool() for tool in self.tools]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=lc_tools,
            llm=self.llm,
            agent="chat-conversational-react-description",
            memory=self.memory,
            verbose=True
        )
    
    async def process(self, query: str) -> Dict[str, Any]:
        """Process a query
        
        Args:
            query: Query string
            
        Returns:
            Response dictionary
        """
        # Get relevant documents from knowledge base
        docs = self.knowledge_base.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Add context to query
        enhanced_query = f"Based on the following context:\n{context}\n\nUser query: {query}"
        
        # Get agent response
        response = self.agent.run(enhanced_query)
        
        return {
            "query": query,
            "response": response,
            "context_docs": len(docs),
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "timestamp": datetime.datetime.now().isoformat()
        }

class NetworkDesignAgent(OpticsAgent):
    """Agent for optical network design assistance"""
    
    def __init__(self, config: AgentConfig, knowledge_base: VectorKnowledgeBase, 
                 component_library, evaluator, simulator):
        """Initialize the network design agent
        
        Args:
            config: Agent configuration
            knowledge_base: Vector knowledge base
            component_library: Component library
            evaluator: Optical evaluator
            simulator: Network simulator
        """
        # Create design-specific tools
        tools = [
            OpticsTool(
                name="search_components",
                description="Search for components matching criteria",
                func=lambda criteria: component_library.find_transceivers(criteria)
            ),
            OpticsTool(
                name="evaluate_design",
                description="Evaluate a network design",
                func=lambda topology, components, links: evaluator.evaluate_network_design(
                    topology, components, links
                )
            ),
            OpticsTool(
                name="simulate_topology",
                description="Simulate a network topology",
                func=lambda config: simulator.simulate_network(config)
            ),
            OpticsTool(
                name="optimize_components",
                description="Optimize component selection based on requirements",
                func=lambda requirements: evaluator.optimize_component_selection(requirements)
            ),
            OpticsTool(
                name="compare_technologies",
                description="Compare different optical technologies",
                func=lambda technologies: evaluator.compare_optical_technologies(technologies)
            )
        ]
        
        super().__init__(config, knowledge_base, tools)
        
        # Store references to components
        self.component_library = component_library
        self.evaluator = evaluator
        self.simulator = simulator
    
    async def design_network(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design a network based on requirements
        
        Args:
            requirements: Network requirements
            
        Returns:
            Network design details
        """
        query = f"""
        Design an optical network with the following requirements:
        
        {json.dumps(requirements, indent=2)}
        
        Consider factors like topology, component selection, power budget, and reliability.
        Provide specific recommendations for transceivers, switches, and connectivity.
        """
        
        response = await self.process(query)
        
        # Extract design recommendations
        # In a real system, we would parse the agent's response to extract structured data
        
        return {
            **response,
            "requirements": requirements,
            "design_recommendations": response["response"]
        }
    
    async def review_design(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Review an existing network design
        
        Args:
            design: Network design to review
            
        Returns:
            Review comments and suggestions
        """
        query = f"""
        Review the following optical network design:
        
        {json.dumps(design, indent=2)}
        
        Identify any issues, inefficiencies, or risks.
        Suggest specific improvements and alternatives where appropriate.
        Consider factors like reliability, power efficiency, scalability, and cost.
        """
        
        response = await self.process(query)
        
        return {
            **response,
            "design": design,
            "review_comments": response["response"]
        }

class NetworkOptimizationAgent(OpticsAgent):
    """Agent for optimizing optical network designs"""
    
    def __init__(self, config: AgentConfig, knowledge_base: VectorKnowledgeBase,
                 evaluator, simulator):
        """Initialize the network optimization agent
        
        Args:
            config: Agent configuration
            knowledge_base: Vector knowledge base
            evaluator: Optical evaluator
            simulator: Network simulator
        """
        # Create optimization-specific tools
        tools = [
            OpticsTool(
                name="evaluate_network",
                description="Evaluate a network design",
                func=lambda topology, components, links: evaluator.evaluate_network_design(
                    topology, components, links
                )
            ),
            OpticsTool(
                name="simulate_performance",
                description="Simulate network performance",
                func=lambda config: simulator.simulate_network(config)
            ),
            OpticsTool(
                name="compare_topologies",
                description="Compare different network topologies",
                func=lambda configs: simulator.compare_topologies(configs)
            ),
            OpticsTool(
                name="optimize_component_selection",
                description="Optimize component selection based on requirements",
                func=lambda requirements: evaluator.optimize_component_selection(requirements)
            )
        ]
        
        super().__init__(config, knowledge_base, tools)
        
        # Store references to components
        self.evaluator = evaluator
        self.simulator = simulator
    
    async def optimize_for_objective(self, network_design: Dict[str, Any], 
                                   objective: str) -> Dict[str, Any]:
        """Optimize a network design for a specific objective
        
        Args:
            network_design: Network design to optimize
            objective: Optimization objective (latency, power, reliability, cost)
            
        Returns:
            Optimized network design
        """
        query = f"""
        Optimize the following optical network design for {objective}:
        
        {json.dumps(network_design, indent=2)}
        
        Suggest specific changes to improve {objective} while maintaining acceptable performance
        in other areas. Consider component selection, topology adjustments, and configuration changes.
        """
        
        response = await self.process(query)
        
        return {
            **response,
            "original_design": network_design,
            "optimization_objective": objective,
            "optimized_design": response["response"]
        }
    
    async def find_bottlenecks(self, network_design: Dict[str, Any], 
                             simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find bottlenecks in a network design based on simulation results
        
        Args:
            network_design: Network design
            simulation_results: Simulation results
            
        Returns:
            Identified bottlenecks and remediation suggestions
        """
        query = f"""
        Analyze the following network design and simulation results to identify bottlenecks:
        
        Network Design:
        {json.dumps(network_design, indent=2)}
        
        Simulation Results:
        {json.dumps(simulation_results, indent=2)}
        
        Identify specific bottlenecks, their causes, and potential remediation strategies.
        Prioritize bottlenecks based on their impact on overall network performance.
        """
        
        response = await self.process(query)
        
        return {
            **response,
            "network_design": network_design,
            "simulation_results": simulation_results,
            "bottlenecks": response["response"]
        }

class FailureAnalysisAgent(OpticsAgent):
    """Agent for analyzing and predicting network failures"""
    
    def __init__(self, config: AgentConfig, knowledge_base: VectorKnowledgeBase,
                 simulator, tester):
        """Initialize the failure analysis agent
        
        Args:
            config: Agent configuration
            knowledge_base: Vector knowledge base
            simulator: Network simulator
            tester: Optical tester
        """
        # Create failure analysis-specific tools
        tools = [
            OpticsTool(
                name="simulate_failures",
                description="Simulate network failure scenarios",
                func=lambda config, scenarios: simulator.simulate_failure_scenarios(
                    config, scenarios
                )
            ),
            OpticsTool(
                name="analyze_test_results",
                description="Analyze component test results",
                func=lambda component_id: tester.analyze_test_results(component_id)
            ),
            OpticsTool(
                name="compare_components",
                description="Compare test results for multiple components",
                func=lambda component_ids: tester.compare_components(component_ids)
            )
        ]
        
        super().__init__(config, knowledge_base, tools)
        
        # Store references to components
        self.simulator = simulator
        self.tester = tester
    
    async def analyze_failure(self, failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a network failure
        
        Args:
            failure_data: Failure data
            
        Returns:
            Failure analysis
        """
        query = f"""
        Analyze the following network failure:
        
        {json.dumps(failure_data, indent=2)}
        
        Identify the root cause, contributing factors, and propagation path.
        Suggest mitigation strategies to prevent similar failures in the future.
        Recommend immediate actions to resolve the current failure.
        """
        
        response = await self.process(query)
        
        return {
            **response,
            "failure_data": failure_data,
            "root_cause_analysis": response["response"]
        }
    
    async def predict_failures(self, network_design: Dict[str, Any], 
                             test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict potential failures in a network design
        
        Args:
            network_design: Network design
            test_results: Component test results
            
        Returns:
            Failure predictions and preventive recommendations
        """
        query = f"""
        Predict potential failures in the following network design based on test results:
        
        Network Design:
        {json.dumps(network_design, indent=2)}
        
        Test Results:
        {json.dumps(test_results, indent=2)}
        
        Identify high-risk components, potential failure modes, and estimated time frames.
        Suggest preventive maintenance and architectural improvements to reduce failure risk.
        Prioritize risks based on impact and probability.
        """
        
        response = await self.process(query)
        
        return {
            **response,
            "network_design": network_design,
            "test_results": test_results,
            "failure_predictions": response["response"]
        }

class AgentManager:
    """Manages the autonomous agents in the system"""
    
    def __init__(self, component_library, evaluator, simulator, tester):
        """Initialize the agent manager
        
        Args:
            component_library: Component library
            evaluator: Optical evaluator
            simulator: Network simulator
            tester: Optical tester
        """
        self.component_library = component_library
        self.evaluator = evaluator
        self.simulator = simulator
        self.tester = tester
        
        # Initialize AI components
        self._initialize_ai_components()
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_ai_components(self):
        """Initialize AI components"""
        # Initialize knowledge base
        self.knowledge_base = VectorKnowledgeBase(
            embedding_model=CONFIG["ai"]["embedding_model"],
            persist_directory=str(KNOWLEDGE_DIR / "vector_store")
        )
        
        # Initialize knowledge graph
        kg_config = CONFIG["knowledge_graph"]
        self.knowledge_graph = KnowledgeGraphManager(
            uri=kg_config["neo4j_uri"],
            user=kg_config["neo4j_user"],
            password=kg_config["neo4j_password"]
        )
        
        # Load initial knowledge
        self._load_initial_knowledge()
    
    def _load_initial_knowledge(self):
        """Load initial knowledge into the knowledge base"""
        # In a production system, we would load domain knowledge from various sources
        # For this example, we'll add some basic optical networking knowledge
        
        basic_knowledge = [
            Document(
                page_content="""
                Optical transceivers are communication devices that transmit and receive information as optical signals.
                The key metrics for transceivers include power budget, reach, data rate, and form factor.
                Common form factors include QSFP, QSFP28, QSFP-DD, and OSFP.
                The power budget is the difference between transmitter output power and receiver sensitivity,
                which determines the maximum link distance.
                """,
                metadata={"source": "domain_knowledge", "topic": "transceivers"}
            ),
            Document(
                page_content="""
                Network topologies define how switches, servers, and other devices are connected.
                Common data center topologies include leaf-spine, Clos, dragonfly, and fat tree.
                Leaf-spine is a two-tier network where spine switches connect to all leaf switches.
                Clos networks are multi-tier networks that provide multiple paths between any two edge devices.
                Topology selection affects scalability, latency, oversubscription, and fault tolerance.
                """,
                metadata={"source": "domain_knowledge", "topic": "topologies"}
            ),
            Document(
                page_content="""
                Optical technologies include pluggables, co-packaged optics, silicon photonics, and coherent optics.
                Pluggable transceivers are the most common and mature technology.
                Co-packaged optics integrate optical components directly with switching ASICs to improve power efficiency.
                Silicon photonics uses silicon as the optical medium for components like modulators and photodetectors.
                Coherent optics are used for long-distance communication and allow higher data rates over longer distances.
                """,
                metadata={"source": "domain_knowledge", "topic": "technologies"}
            ),
            Document(
                page_content="""
                Failure modes in optical networks include laser degradation, receiver sensitivity drift,
                thermal drift, link flaps, mechanical damage, and contamination.
                Laser degradation occurs over time as the laser diode ages, reducing output power.
                Receiver sensitivity drift can occur due to temperature changes or aging.
                Link flaps are rapid transitions between up and down states, often caused by marginal power levels.
                Contamination of fiber connectors can significantly increase insertion loss.
                """,
                metadata={"source": "domain_knowledge", "topic": "failures"}
            )
        ]
        
        self.knowledge_base.add_documents(basic_knowledge)
    
    def _initialize_agents(self):
        """Initialize all agents"""
        # Create AI model config
        ai_config = CONFIG["ai"]
        model_config = AIModelConfig(
            model_name=ai_config["model"],
            temperature=ai_config["temperature"],
            max_tokens=ai_config["max_tokens"],
            api_key=ai_config["openai_api_key"],
            embedding_model=ai_config["embedding_model"]
        )
        
        # Initialize design assistant agent
        if CONFIG["agents"]["enable_design_assistant_agent"]:
            design_config = AgentConfig(
                agent_id="design_assistant",
                agent_type="network_design",
                enabled=True,
                check_interval=CONFIG["agents"]["agent_check_interval"],
                tools=["search_components", "evaluate_design", "simulate_topology", 
                       "optimize_components", "compare_technologies"],
                llm_config=model_config,
                memory_size=10
            )
            
            self.agents["design_assistant"] = NetworkDesignAgent(
                config=design_config,
                knowledge_base=self.knowledge_base,
                component_library=self.component_library,
                evaluator=self.evaluator,
                simulator=self.simulator
            )
        
        # Initialize optimization agent
        if CONFIG["agents"]["enable_optimization_agent"]:
            optimization_config = AgentConfig(
                agent_id="network_optimizer",
                agent_type="network_optimization",
                enabled=True,
                check_interval=CONFIG["agents"]["agent_check_interval"],
                tools=["evaluate_network", "simulate_performance", "compare_topologies", 
                       "optimize_component_selection"],
                llm_config=model_config,
                memory_size=10
            )
            
            self.agents["network_optimizer"] = NetworkOptimizationAgent(
                config=optimization_config,
                knowledge_base=self.knowledge_base,
                evaluator=self.evaluator,
                simulator=self.simulator
            )
        
        # Initialize failure analysis agent
        if CONFIG["agents"]["enable_failure_analysis_agent"]:
            failure_config = AgentConfig(
                agent_id="failure_analyst",
                agent_type="failure_analysis",
                enabled=True,
                check_interval=CONFIG["agents"]["agent_check_interval"],
                tools=["simulate_failures", "analyze_test_results", "compare_components"],
                llm_config=model_config,
                memory_size=10
            )
            
            self.agents["failure_analyst"] = FailureAnalysisAgent(
                config=failure_config,
                knowledge_base=self.knowledge_base,
                simulator=self.simulator,
                tester=self.tester
            )
    
    async def process_query(self, agent_id: str, query: str) -> Dict[str, Any]:
        """Process a query with a specific agent
        
        Args:
            agent_id: Agent ID
            query: Query string
            
        Returns:
            Agent response
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        agent = self.agents[agent_id]
        return await agent.process(query)
    
    async def design_network(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design a network based on requirements
        
        Args:
            requirements: Network requirements
            
        Returns:
            Network design details
        """
        if "design_assistant" not in self.agents:
            raise ValueError("Design assistant agent not available")
        
        agent = self.agents["design_assistant"]
        return await agent.design_network(requirements)
    
    async def optimize_network(self, network_design: Dict[str, Any], 
                             objective: str) -> Dict[str, Any]:
        """Optimize a network design for a specific objective
        
        Args:
            network_design: Network design to optimize
            objective: Optimization objective
            
        Returns:
            Optimized network design
        """
        if "network_optimizer" not in self.agents:
            raise ValueError("Network optimizer agent not available")
        
        agent = self.agents["network_optimizer"]
        return await agent.optimize_for_objective(network_design, objective)
    
    async def analyze_failure(self, failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a network failure
        
        Args:
            failure_data: Failure data
            
        Returns:
            Failure analysis
        """
        if "failure_analyst" not in self.agents:
            raise ValueError("Failure analyst agent not available")
        
        agent = self.agents["failure_analyst"]
        return await agent.analyze_failure(failure_data)
    
    def update_knowledge(self, component: Any):
        """Update knowledge base with component information
        
        Args:
            component: Component to add to knowledge base
        """
        # Update knowledge graph
        self.knowledge_graph.add_component(component)
        
        # Update vector knowledge base
        component_dict = component.to_dict()
        component_str = f"""
        Component ID: {component_dict.get('id')}
        Name: {component_dict.get('name')}
        Type: {component_dict.get('component_type')}
        Manufacturer: {component_dict.get('manufacturer')}
        Model: {component_dict.get('model')}
        Data Rate: {component_dict.get('data_rate')}
        """
        
        # Add specific details based on component type
        if hasattr(component, "technology"):
            component_str += f"""
            Technology: {component_dict.get('technology')}
            Wavelength Technology: {component_dict.get('wavelength_tech')}
            Form Factor: {component_dict.get('form_factor')}
            Reach: {component_dict.get('reach')} m
            TX Power: {component_dict.get('tx_power')} dBm
            RX Sensitivity: {component_dict.get('rx_sensitivity')} dBm
            """
        
        self.knowledge_base.add_texts([component_str], [{"component_id": component_dict.get('id')}])

#############################################################
# AI-Enhanced API Implementation
#############################################################

class OpticalNetworkAI:
    """AI-powered API for optical network design and evaluation"""
    
    def __init__(self):
        """Initialize the API with AI capabilities"""
        self.component_library = ComponentLibrary()
        self.evaluator = OpticalEvaluator(self.component_library)
        self.simulator = NetworkSimulator()
        self.tester = OpticalTester(self.component_library)
        
        # Initialize AI components
        self.agent_manager = AgentManager(
            self.component_library,
            self.evaluator,
            self.simulator,
            self.tester
        )
        
        # Create FastAPI app
        self.app = FastAPI(
            title="OpenOptics AI",
            description="AI-Enhanced Optical Network Architecture API",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=CONFIG["api"]["cors"]["allowed_origins"],
            allow_credentials=True,
            allow_methods=CONFIG["api"]["cors"]["allowed_methods"],
            allow_headers=CONFIG["api"]["cors"]["allowed_headers"],
        )
        
        # Serve static files
        self.app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        app = self.app
        
        # Original routes from the base implementation...
        # Adding some example non-AI routes for core functionalities

        @app.get("/components")
        async def list_components():
            """List all available components from the library."""
            # This is a simplified example. In a real app, you might want pagination, filtering, etc.
            # And ComponentLibrary would need a method to list all components.
            # For now, returning the internal dictionary directly (not ideal for production).
            if hasattr(self.component_library, 'components'): # Check if the placeholder attribute exists
                return self.component_library.components
            elif hasattr(self.component_library, 'get_all_components'): # Hypothetical better method
                return await self.component_library.get_all_components()
            return {"message": "Component listing not fully implemented in placeholder."}

        @app.post("/evaluate_design")
        async def evaluate_design_direct(design_data: Dict[str, Any]): # Define a Pydantic model for better validation
            """Evaluate a network design directly."""
            # Expects design_data to have topology, components (list of IDs or dicts), links
            components_data = []
            for comp_ref in design_data.get("components", []):
                if isinstance(comp_ref, str):
                    comp = self.component_library.get_component(comp_ref)
                    if comp: components_data.append(comp)
                    else: raise HTTPException(status_code=404, detail=f"Component {comp_ref} not found.")
                elif isinstance(comp_ref, dict):
                    components_data.append(comp_ref)
            
            try:
                result = self.evaluator.evaluate_network_design(
                    design_data.get("topology", {}),
                    components_data,
                    design_data.get("links", [])
                )
                return result
            except Exception as e:
                logger.error(f"Error in direct design evaluation: {e}")
                raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

        @app.post("/simulate_network")
        async def simulate_network_direct(simulation_config: Dict[str, Any]): # Define Pydantic model
            """Run a network simulation directly."""
            try:
                result = self.simulator.simulate_network(simulation_config)
                return result
            except Exception as e:
                logger.error(f"Error in direct simulation: {e}")
                raise HTTPException(status_code=500, detail=f"Error during simulation: {str(e)}")

        @app.get("/test_component/{component_id}")
        async def test_component_direct(component_id: str):
            """Get test analysis for a specific component directly."""
            try:
                # Check if component exists in library first, if desired
                # comp = self.component_library.get_component(component_id)
                # if not comp:
                #     raise HTTPException(status_code=404, detail=f"Component {component_id} not found in library.")
                result = self.tester.analyze_test_results(component_id)
                return result
            except Exception as e:
                logger.error(f"Error in direct component test analysis: {e}")
                raise HTTPException(status_code=500, detail=f"Error during test analysis: {str(e)}")

        # AI-powered routes
        @app.post("/ai/design")
        async def design_network(requirements: Dict[str, Any]):
            """Design a network based on requirements"""
            result = await self.agent_manager.design_network(requirements)
            return result
        
        @app.post("/ai/optimize")
        async def optimize_network(network_design: Dict[str, Any], objective: str):
            """Optimize a network design for a specific objective"""
            result = await self.agent_manager.optimize_network(network_design, objective)
            return result
        
        @app.post("/ai/analyze_failure")
        async def analyze_failure(failure_data: Dict[str, Any]):
            """Analyze a network failure"""
            result = await self.agent_manager.analyze_failure(failure_data)
            return result
        
        @app.post("/ai/query/{agent_id}")
        async def query_agent(agent_id: str, query: str):
            """Query a specific agent"""
            try:
                result = await self.agent_manager.process_query(agent_id, query)
                return result
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time interactions"""
            await websocket.accept()
            
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_json()
                    
                    # Extract relevant fields
                    agent_id = data.get("agent_id", "design_assistant")
                    query = data.get("query", "")
                    
                    if not query:
                        await websocket.send_json({"error": "Query is required"})
                        continue
                    
                    # Process query with appropriate agent
                    try:
                        result = await self.agent_manager.process_query(agent_id, query)
                        await websocket.send_json(result)
                    except ValueError as e:
                        await websocket.send_json({"error": str(e)})
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        await websocket.send_json({"error": "Internal server error"})
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
        
        @app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the main application page"""
            with open(BASE_DIR / "static" / "index.html", "r") as f:
                return f.read()
    
    def run(self, host: str = None, port: int = None):
        """Run the API server"""
        host = host or CONFIG["api"]["host"]
        port = port or CONFIG["api"]["port"]
        
        uvicorn.run(self.app, host=host, port=port)

#############################################################
# Main Entry Point
#############################################################

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="OpenOptics: AI-Enhanced Optical Network Framework")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['api', 'evaluate', 'simulate', 'test', 'ai'],
                      default='api', help='Mode to run')
    
    # API mode arguments
    parser.add_argument('--host', type=str, help='Host for API server')
    parser.add_argument('--port', type=int, help='Port for API server')
    
    # AI mode arguments
    parser.add_argument('--query', type=str, help='Query for AI agent')
    parser.add_argument('--agent', type=str, default='design_assistant', 
                      help='Agent to use (design_assistant, network_optimizer, failure_analyst)')
    
    # Arguments for other modes
    parser.add_argument('--input_file', type=str, help='Path to input JSON file for evaluate/simulate modes')
    parser.add_argument('--component_id', type=str, help='Component ID for test mode')

    args = parser.parse_args()
    
    # Initialize components (used by multiple modes)
    component_library = ComponentLibrary()
    evaluator = OpticalEvaluator(component_library)
    simulator = NetworkSimulator()
    tester = OpticalTester(component_library)

    if args.mode == 'api':
        # Run API server with AI capabilities
        # AgentManager is initialized within OpticalNetworkAI
        api = OpticalNetworkAI() # This already initializes AgentManager with the components above
        api.run(host=args.host, port=args.port)
    
    elif args.mode == 'ai':
        # Run AI query directly
        if not args.query:
            print("Please specify a query with --query")
            return
        
        # AgentManager needs to be initialized for AI mode if not running API
        agent_manager = AgentManager(component_library, evaluator, simulator, tester)
        
        # Run query
        # import asyncio # Already imported globally
        result = asyncio.run(agent_manager.process_query(args.agent, args.query))
        print(json.dumps(result, indent=2))

    elif args.mode == 'evaluate':
        print("Running in evaluation mode...")
        if not args.input_file:
            print("Please specify an input file for evaluation with --input_file.")
            # Providing a default dummy evaluation call if no file
            print("Performing a dummy evaluation as no input file was provided.")
            dummy_topology = {"name": "dummy_topo"} # Placeholder
            dummy_components = [component_library.get_component("trx100G")] # Placeholder
            dummy_links = [{"source": "trx100G", "target": "switch32p"}] # Placeholder
            evaluation_result = evaluator.evaluate_network_design(dummy_topology, dummy_components, dummy_links)
        else:
            try:
                with open(args.input_file, 'r') as f:
                    design_data = json.load(f)
                # Assuming design_data has keys: topology, components, links
                # Components might be list of IDs, need to fetch from library
                components_data = []
                for comp_ref in design_data.get("components", []):
                    if isinstance(comp_ref, str):
                        comp = component_library.get_component(comp_ref)
                        if comp: components_data.append(comp)
                    elif isinstance(comp_ref, dict):
                         components_data.append(comp_ref) # Assume full component data
                
                evaluation_result = evaluator.evaluate_network_design(
                    design_data.get("topology", {}),
                    components_data,
                    design_data.get("links", [])
                )
            except FileNotFoundError:
                print(f"Error: Input file {args.input_file} not found.")
                return
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {args.input_file}.")
                return
        print(json.dumps(evaluation_result, indent=2))

    elif args.mode == 'simulate':
        print("Running in simulation mode...")
        if not args.input_file:
            print("Please specify an input file for simulation with --input_file (network config).")
            print("Performing a dummy simulation as no input file was provided.")
            dummy_sim_config = {"name": "dummy_simulation", "duration_hours": 1} # Placeholder
            simulation_result = simulator.simulate_network(dummy_sim_config)
        else:
            try:
                with open(args.input_file, 'r') as f:
                    sim_config = json.load(f)
                simulation_result = simulator.simulate_network(sim_config)
            except FileNotFoundError:
                print(f"Error: Input file {args.input_file} not found.")
                return
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {args.input_file}.")
                return
        print(json.dumps(simulation_result, indent=2))

    elif args.mode == 'test':
        print("Running in testing mode...")
        if not args.component_id:
            print("Please specify a component ID with --component_id.")
            print("Testing a dummy component 'trx100G' as no ID was provided.")
            component_id_to_test = "trx100G"
        else:
            component_id_to_test = args.component_id
        
        test_result = tester.analyze_test_results(component_id_to_test)
        print(json.dumps(test_result, indent=2))
    
    # Other modes (evaluate, simulate, test) can be implemented as in the original code
    # Basic implementations added above.

if __name__ == "__main__":
    main()