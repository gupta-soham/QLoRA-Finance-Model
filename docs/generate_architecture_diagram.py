#!/usr/bin/env python3
"""
Financial QLoRA Fine-tuning Architecture Diagram Generator

This script generates architecture diagrams for the Financial QLoRA fine-tuning system
using the diagrams package with official AWS icons.

Usage:
    python generate_architecture_diagram.py

Requirements:
    pip install diagrams
"""

from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2, EC2Instance, Lambda
from diagrams.aws.database import RDS
from diagrams.aws.storage import S3
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerNotebook, DLC
from diagrams.aws.network import APIGateway
from diagrams.aws.database import ElastiCache
from diagrams.aws.management import CloudwatchAlarm
from diagrams.aws.general import User
from diagrams.generic.os import Windows


def generate_png_diagram():
    """Generate PNG format diagram"""
    with Diagram("Financial QLoRA Fine-tuning Architecture", 
                 show=False, 
                 direction="LR", 
                 graph_attr={"splines": "ortho", "nodesep": "2.0", "ranksep": "2.5", "pad": "1.0"},
                 filename="financial_qlora_architecture"):
        
        with Cluster("Development", graph_attr={"margin": "30"}):
            user = User("Developer")
            windows = Windows("Windows")
            
        with Cluster("Data Sources", graph_attr={"margin": "30"}):
            nifty = RDS("NIFTY")
            sensex = RDS("SENSEX") 
            stocks = RDS("Stocks")
            
        with Cluster("Training", graph_attr={"margin": "30"}):
            dataset = S3("Dataset")
            model = Sagemaker("TinyLlama")
            qlora = EC2("QLoRA")
            
        with Cluster("Infrastructure", graph_attr={"margin": "30"}):
            gpu = EC2Instance("GPU")
            pytorch = DLC("PyTorch")
            optimizer = Lambda("Optimizer")
            
        with Cluster("Model", graph_attr={"margin": "30"}):
            peft = SagemakerModel("PEFT")
            template = S3("Template")
            tokenizer = S3("Tokenizer")
            
        with Cluster("Process", graph_attr={"margin": "30"}):
            train = S3("Train")
            val = S3("Validation")
            stop = CloudwatchAlarm("Monitor")
            
        with Cluster("Output", graph_attr={"margin": "30"}):
            final = S3("Model")
            inference = SagemakerNotebook("Inference")
            
        with Cluster("Live Data", graph_attr={"margin": "30"}):
            api = APIGateway("API")
            context = ElastiCache("Context")
            
        # Connections
        user >> windows >> dataset
        nifty >> dataset
        sensex >> dataset
        stocks >> dataset
        dataset >> model >> qlora >> peft
        gpu >> pytorch >> optimizer >> qlora
        peft >> template >> tokenizer
        dataset >> train >> stop
        dataset >> val >> stop
        stop >> final >> inference
        api >> context >> inference
        user >> inference


def generate_svg_diagram():
    """Generate SVG format diagram"""
    with Diagram("Financial QLoRA Fine-tuning Architecture", 
                 show=False, 
                 direction="LR", 
                 graph_attr={"splines": "ortho", "nodesep": "2.0", "ranksep": "2.5", "pad": "1.0"},
                 outformat="svg",
                 filename="financial_qlora_architecture_svg"):
        
        with Cluster("Development", graph_attr={"margin": "30"}):
            user = User("Developer")
            windows = Windows("Windows")
            
        with Cluster("Data Sources", graph_attr={"margin": "30"}):
            nifty = RDS("NIFTY")
            sensex = RDS("SENSEX") 
            stocks = RDS("Stocks")
            
        with Cluster("Training", graph_attr={"margin": "30"}):
            dataset = S3("Dataset")
            model = Sagemaker("TinyLlama")
            qlora = EC2("QLoRA")
            
        with Cluster("Infrastructure", graph_attr={"margin": "30"}):
            gpu = EC2Instance("GPU")
            pytorch = DLC("PyTorch")
            optimizer = Lambda("Optimizer")
            
        with Cluster("Model", graph_attr={"margin": "30"}):
            peft = SagemakerModel("PEFT")
            template = S3("Template")
            tokenizer = S3("Tokenizer")
            
        with Cluster("Process", graph_attr={"margin": "30"}):
            train = S3("Train")
            val = S3("Validation")
            stop = CloudwatchAlarm("Monitor")
            
        with Cluster("Output", graph_attr={"margin": "30"}):
            final = S3("Model")
            inference = SagemakerNotebook("Inference")
            
        with Cluster("Live Data", graph_attr={"margin": "30"}):
            api = APIGateway("API")
            context = ElastiCache("Context")
            
        # Connections
        user >> windows >> dataset
        nifty >> dataset
        sensex >> dataset
        stocks >> dataset
        dataset >> model >> qlora >> peft
        gpu >> pytorch >> optimizer >> qlora
        peft >> template >> tokenizer
        dataset >> train >> stop
        dataset >> val >> stop
        stop >> final >> inference
        api >> context >> inference
        user >> inference


if __name__ == "__main__":
    print("Generating Financial QLoRA Architecture Diagrams...")
    
    print("Creating PNG diagram...")
    generate_png_diagram()
    print("✓ PNG diagram saved as: financial_qlora_architecture.png")
    
    print("Creating SVG diagram...")
    generate_svg_diagram()
    print("✓ SVG diagram saved as: financial_qlora_architecture_svg.svg")
    
    print("Done! Both diagrams generated successfully.")
