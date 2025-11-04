# FlyingCarRL Cloud Deployment

This directory contains deployment scripts and configurations for running FlyingCarRL in cloud environments.

## Deployment Options

### 1. Docker (Local or Any Cloud)

The simplest way to deploy FlyingCarRL is using Docker.

#### Prerequisites
- Docker installed
- Docker Compose installed
- NVIDIA Docker runtime (for GPU support)

#### Quick Start

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

#### Services Included
- **simulator**: Main simulation service (headless)
- **trainer**: Training service with RL algorithms
- **dashboard**: Streamlit web dashboard (port 8501)
- **mysql**: MySQL database for logging
- **ray-head**: Ray cluster head for distributed training (port 8265)

#### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DB_TYPE=mysql
DB_HOST=mysql
DB_PORT=3306
DB_NAME=flyingcarrl
DB_USER=flyingcar
DB_PASSWORD=your_secure_password

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key

# NVIDIA GPU
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

#### Custom Deployments

Run specific services:

```bash
# Training only
docker-compose up -d trainer mysql

# Dashboard only
docker-compose up -d dashboard mysql

# Distributed training
docker-compose up -d ray-head trainer mysql
```

### 2. AWS EC2 Deployment

Deploy FlyingCarRL to AWS EC2 with GPU instances.

#### Prerequisites
- AWS CLI installed and configured
- AWS account with EC2 permissions
- Sufficient EC2 GPU instance quota

#### Deploy

```bash
cd deploy/aws

# Configure (optional)
export INSTANCE_TYPE="g4dn.xlarge"  # or g4dn.2xlarge, p3.2xlarge, etc.
export REGION="us-east-1"
export KEY_NAME="flyingcarrl-key"

# Deploy
bash deploy_ec2.sh
```

#### What Gets Created
- EC2 GPU instance (default: g4dn.xlarge)
- Security group with ports 22, 8501, 8265
- SSH key pair (saved as `flyingcarrl-key.pem`)
- Docker environment with all services

#### Access

After deployment completes (5-10 minutes):

```bash
# SSH to instance
ssh -i flyingcarrl-key.pem ubuntu@<INSTANCE_IP>

# View services
docker-compose ps

# View logs
docker-compose logs -f simulator
```

#### Dashboard Access
- Streamlit: `http://<INSTANCE_IP>:8501`
- Ray Dashboard: `http://<INSTANCE_IP>:8265`

#### Costs
- g4dn.xlarge: ~$0.526/hour (1 NVIDIA T4 GPU)
- g4dn.2xlarge: ~$0.752/hour (1 NVIDIA T4 GPU, more CPU/RAM)
- p3.2xlarge: ~$3.06/hour (1 NVIDIA V100 GPU)

Remember to stop or terminate instances when not in use!

#### Cleanup

```bash
# Find instance ID
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=FlyingCarRL-Training" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text

# Stop instance (keeps data)
aws ec2 stop-instances --instance-ids <INSTANCE_ID>

# Terminate instance (deletes data)
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

### 3. Google Cloud Platform (GCP) Deployment

Deploy FlyingCarRL to Google Compute Engine with GPU support.

#### Prerequisites
- gcloud CLI installed and configured
- GCP account with Compute Engine API enabled
- GPU quota in desired region

#### Deploy

```bash
cd deploy/gcp

# Configure project
gcloud config set project YOUR_PROJECT_ID

# Optional: Configure settings
export INSTANCE_NAME="flyingcarrl-training"
export MACHINE_TYPE="n1-standard-4"
export ACCELERATOR_TYPE="nvidia-tesla-t4"
export ZONE="us-central1-a"

# Deploy
bash deploy_gce.sh
```

#### What Gets Created
- GCE instance with GPU (default: n1-standard-4 + T4)
- Firewall rules for ports 8501, 8265
- Deep Learning VM image with CUDA pre-installed
- Docker environment with all services

#### Access

After deployment completes:

```bash
# SSH to instance
gcloud compute ssh flyingcarrl-training --zone=us-central1-a

# View services
docker ps

# View logs
docker-compose logs -f
```

#### Dashboard Access
- Streamlit: `http://<EXTERNAL_IP>:8501`
- Ray Dashboard: `http://<EXTERNAL_IP>:8265`

#### Costs
- n1-standard-4 + T4: ~$0.67/hour
- n1-standard-8 + T4: ~$0.95/hour
- n1-standard-4 + V100: ~$2.90/hour

#### Cleanup

```bash
# Stop instance (keeps data)
gcloud compute instances stop flyingcarrl-training --zone=us-central1-a

# Delete instance (removes data)
gcloud compute instances delete flyingcarrl-training --zone=us-central1-a

# Delete firewall rules (optional)
gcloud compute firewall-rules delete flyingcarrl-allow-dashboard
gcloud compute firewall-rules delete flyingcarrl-allow-ray
```

## Advanced Configurations

### Multi-GPU Training

For multiple GPUs, modify `docker-compose.yml`:

```yaml
trainer:
  environment:
    - NVIDIA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3
  command: python scripts/distributed_training.py --num-gpus 4 --num-workers 16
```

### Distributed Multi-Node Training

For multi-node Ray clusters:

1. Start Ray head node:
```bash
docker-compose up -d ray-head
```

2. On worker nodes:
```bash
docker run --runtime=nvidia --rm -it flyingcarrl:latest \
    ray start --address=<HEAD_IP>:6379 --block
```

3. Connect and train:
```bash
python scripts/distributed_training.py \
    --ray-address "ray://<HEAD_IP>:10001" \
    --num-workers 32
```

### Custom Resource Allocation

Modify `docker-compose.yml` to limit resources:

```yaml
trainer:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 16G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Persistent Storage

For long-term storage, use cloud volumes:

**AWS EBS:**
```bash
# Create and attach EBS volume
aws ec2 create-volume --size 500 --volume-type gp3 --availability-zone us-east-1a
aws ec2 attach-volume --volume-id vol-xxx --instance-id i-xxx --device /dev/sdf
```

**GCP Persistent Disk:**
```bash
# Create and attach disk
gcloud compute disks create flyingcarrl-data --size=500GB --zone=us-central1-a
gcloud compute instances attach-disk flyingcarrl-training --disk=flyingcarrl-data --zone=us-central1-a
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check all services
docker-compose ps

# Check specific service
docker-compose logs simulator

# Follow logs in real-time
docker-compose logs -f trainer

# Resource usage
docker stats
```

### Database Backup

```bash
# Backup MySQL database
docker-compose exec mysql mysqldump -u flyingcar -p flyingcarrl > backup.sql

# Restore database
docker-compose exec -T mysql mysql -u flyingcar -p flyingcarrl < backup.sql
```

### Updates

```bash
# Pull latest code
git pull origin main

# Rebuild images
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker-compose exec trainer nvidia-smi
```

### Out of Memory

Reduce batch sizes or number of parallel agents:

```bash
docker-compose exec trainer python main.py --mode=headless --agents=10
```

### Connection Issues

Check firewall rules allow ports 8501, 8265:

```bash
# AWS
aws ec2 describe-security-groups --group-names flyingcarrl-sg

# GCP
gcloud compute firewall-rules list --filter="name~flyingcarrl"
```

## Security Best Practices

1. **Change default passwords** in `.env` and `docker-compose.yml`
2. **Restrict security group access** to specific IP ranges
3. **Use IAM roles** instead of hardcoded credentials
4. **Enable SSL/TLS** for dashboard access
5. **Regularly update** Docker images and dependencies
6. **Monitor costs** and set billing alerts
7. **Use secrets management** (AWS Secrets Manager, GCP Secret Manager)

## Cost Optimization

1. **Use spot/preemptible instances** (up to 90% savings)
2. **Stop instances** when not training
3. **Use smaller instances** for development
4. **Enable auto-shutdown** after idle period
5. **Use S3/GCS** for checkpoints instead of EBS/PD
6. **Monitor and set budgets**

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/FlyingCarRL/issues
- Documentation: See `docs/` directory
- Community: [Link to Discord/Forum]
