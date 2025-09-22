# Animal Classification AWS Infrastructure

This directory contains Terraform configuration to deploy the animal classification application on AWS using ECS Fargate.

## Architecture

- **ECR Repository**: Docker registry for the application image
- **ECS Cluster**: Fargate cluster to run containers
- **ECS Service**: Manages application instances with public IPs
- **VPC**: Isolated network with public/private subnets
- **Security Groups**: Network security for ECS tasks
- **IAM Roles**: Required permissions for ECS tasks

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Docker image built and ready to push to ECR

## Deployment Steps

1. **Initialize Terraform**
   ```bash
   cd infra/aws
   terraform init
   ```

2. **Configure Variables**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your preferred values
   ```

3. **Plan Deployment**
   ```bash
   terraform plan
   ```

4. **Deploy Infrastructure**
   ```bash
   terraform apply
   ```

5. **Build and Push Docker Image**
   ```bash
   # Get ECR login command
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build image from project root
   cd ../..
   docker build -f infra/Dockerfile -t animal-classification .

   # Tag and push image
   docker tag animal-classification:latest <ecr-repository-url>:latest
   docker push <ecr-repository-url>:latest
   ```

6. **Update ECS Service**
   ```bash
   aws ecs update-service --cluster animal-classification-cluster --service animal-classification-service --force-new-deployment
   ```

## Accessing the Application

After deployment, the application will be available on the public IPs of the ECS tasks on port 8000.

To get the public IPs of running tasks:

1. **List running tasks:**
   ```bash
   aws ecs list-tasks --cluster animal-classification-cluster --service-name animal-classification-service
   ```

2. **Get task details including public IP:**
   ```bash
   aws ecs describe-tasks --cluster animal-classification-cluster --tasks <task-arn>
   ```

3. **Access the application:**
   ```
   http://<public-ip>:8000
   ```

## Monitoring

- CloudWatch logs are available in the `/ecs/animal-classification` log group
- ECS service metrics are available in CloudWatch
- Container Insights are enabled for detailed monitoring

## Cleanup

To destroy all resources:
```bash
terraform destroy
```

## Cost Optimization

The current configuration uses:
- 2 Fargate tasks (1024 CPU, 2048 MB memory each)
- NAT Gateways (2)

For development/testing, consider:
- Reducing to 1 Fargate task
- Using smaller CPU/memory allocations
- Using a single NAT Gateway