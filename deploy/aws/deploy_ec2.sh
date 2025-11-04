#!/bin/bash
# Deploy FlyingCarRL to AWS EC2 with GPU support
# This script provisions an EC2 instance and deploys the application

set -e

# Configuration
INSTANCE_TYPE=${INSTANCE_TYPE:-"g4dn.xlarge"}  # GPU instance
AMI_ID=${AMI_ID:-"ami-0c55b159cbfafe1f0"}  # Deep Learning AMI (Ubuntu)
KEY_NAME=${KEY_NAME:-"flyingcarrl-key"}
SECURITY_GROUP=${SECURITY_GROUP:-"flyingcarrl-sg"}
REGION=${REGION:-"us-east-1"}
INSTANCE_NAME="FlyingCarRL-Training"

echo "=========================================="
echo "FlyingCarRL AWS EC2 Deployment"
echo "=========================================="
echo "Instance Type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo "=========================================="

# Check AWS CLI installation
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    echo "Install: pip install awscli"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

# Create security group if it doesn't exist
echo "Creating security group..."
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION &> /dev/null; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "Security group for FlyingCarRL" \
        --region $REGION \
        --query 'GroupId' \
        --output text)

    # Add inbound rules
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region $REGION  # SSH

    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp --port 8501 --cidr 0.0.0.0/0 \
        --region $REGION  # Streamlit

    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp --port 8265 --cidr 0.0.0.0/0 \
        --region $REGION  # Ray dashboard

    echo "Security group created: $SG_ID"
else
    echo "Security group already exists"
fi

# Create key pair if it doesn't exist
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    echo "Creating key pair..."
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --region $REGION \
        --query 'KeyMaterial' \
        --output text > ${KEY_NAME}.pem
    chmod 400 ${KEY_NAME}.pem
    echo "Key pair created and saved to ${KEY_NAME}.pem"
else
    echo "Key pair already exists"
fi

# Create user data script
cat > user_data.sh << 'EOF'
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repository
cd /home/ubuntu
git clone https://github.com/yourusername/FlyingCarRL.git
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

# Start services
docker-compose up -d

echo "FlyingCarRL deployment complete!"
EOF

# Launch EC2 instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --user-data file://user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to start..."

aws ec2 wait instance-running \
    --instance-ids $INSTANCE_ID \
    --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "SSH Command: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "Dashboard: http://${PUBLIC_IP}:8501"
echo "Ray Dashboard: http://${PUBLIC_IP}:8265"
echo "=========================================="
echo ""
echo "Note: It may take 5-10 minutes for the application to fully start"
echo "Check status: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'docker-compose -f /home/ubuntu/FlyingCarRL/docker-compose.yml ps'"

# Cleanup
rm user_data.sh
