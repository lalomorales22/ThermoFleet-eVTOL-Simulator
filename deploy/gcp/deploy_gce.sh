#!/bin/bash
# Deploy FlyingCarRL to Google Cloud Platform (GCP) with GPU support
# This script provisions a GCE instance and deploys the application

set -e

# Configuration
INSTANCE_NAME=${INSTANCE_NAME:-"flyingcarrl-training"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-4"}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-"nvidia-tesla-t4"}
ACCELERATOR_COUNT=${ACCELERATOR_COUNT:-"1"}
ZONE=${ZONE:-"us-central1-a"}
PROJECT_ID=${PROJECT_ID:-""}
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "=========================================="
echo "FlyingCarRL GCP Deployment"
echo "=========================================="
echo "Instance Name: $INSTANCE_NAME"
echo "Machine Type: $MACHINE_TYPE"
echo "Accelerator: $ACCELERATOR_TYPE x$ACCELERATOR_COUNT"
echo "Zone: $ZONE"
echo "=========================================="

# Check gcloud CLI installation
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo "Error: GCP project not set"
        echo "Set project: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
fi

echo "Using project: $PROJECT_ID"

# Create startup script
cat > startup-script.sh << 'EOF'
#!/bin/bash
set -e

# Install NVIDIA drivers and CUDA (already included in Deep Learning image)
echo "Checking NVIDIA drivers..."
nvidia-smi || echo "NVIDIA drivers not found"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
fi

# Install NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Clone repository
cd /home
if [ ! -d "FlyingCarRL" ]; then
    git clone https://github.com/yourusername/FlyingCarRL.git
fi

cd FlyingCarRL

# Create .env file
cat > .env << ENVEOF
DB_TYPE=mysql
DB_HOST=mysql
DB_PORT=3306
DB_NAME=flyingcarrl
DB_USER=flyingcar
DB_PASSWORD=changeme_in_production
ENVEOF

# Pull and start services
docker-compose pull
docker-compose up -d

echo "FlyingCarRL deployment complete!"
EOF

# Create firewall rules
echo "Creating firewall rules..."
gcloud compute firewall-rules create flyingcarrl-allow-dashboard \
    --project=$PROJECT_ID \
    --allow=tcp:8501 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow access to Streamlit dashboard" \
    2>/dev/null || echo "Firewall rule for dashboard already exists"

gcloud compute firewall-rules create flyingcarrl-allow-ray \
    --project=$PROJECT_ID \
    --allow=tcp:8265 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow access to Ray dashboard" \
    2>/dev/null || echo "Firewall rule for Ray already exists"

# Create instance
echo "Creating GCE instance..."
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT \
    --maintenance-policy=TERMINATE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --metadata-from-file=startup-script=startup-script.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=flyingcarrl

echo "Waiting for instance to start..."
sleep 30

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Instance Name: $INSTANCE_NAME"
echo "External IP: $EXTERNAL_IP"
echo "SSH Command: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "Dashboard: http://${EXTERNAL_IP}:8501"
echo "Ray Dashboard: http://${EXTERNAL_IP}:8265"
echo "=========================================="
echo ""
echo "Note: It may take 5-10 minutes for the application to fully start"
echo "Check status: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='docker ps'"

# Cleanup
rm startup-script.sh

# Save instance info
cat > instance_info.txt << EOF
Instance Name: $INSTANCE_NAME
Zone: $ZONE
External IP: $EXTERNAL_IP
Project: $PROJECT_ID

SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
Stop: gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
Start: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
Delete: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
EOF

echo ""
echo "Instance information saved to instance_info.txt"
