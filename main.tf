terraform {
  # This block configures Terraform to store its state file remotely in an S3 bucket.
  # This is a best practice for collaboration and state management.
  backend "s3" {
    bucket = "major1-tfstate-bucket-akash21357"
    key    = "major1/terraform.tfstate"
    region = "ap-south-1"
  }
}

# This block configures the AWS provider, specifying the region where resources will be created.
provider "aws" {
  region = "ap-south-1"
}

# This resource defines the firewall rules (Security Group) for your server.
<<<<<<< HEAD
resource "aws_security_group" "major2357_sg" {
  name        = "major2357-instance-sg"
=======
resource "aws_security_group" "major2_sg" {
  name        = "major2-instance-sg"
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba
  description = "Allow SSH, HTTP, and HTTPS traffic"

  # Allow inbound SSH traffic on port 22 for remote management.
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow inbound HTTP traffic on port 80 for the web server.
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # --- THIS IS THE NEW RULE ---
  # Allow inbound HTTPS traffic on port 443 for the Caddy web server's SSL.
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic from the server.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# This resource defines the EC2 virtual server itself.
<<<<<<< HEAD
resource "aws_instance" "major2357_server" {
  ami           = "ami-0f918f7e67a3323f0"
  instance_type = "t2.micro"
  key_name      = "major2357" # Make sure this matches the name in your AWS Console
  
  # This attaches the security group defined above to the EC2 instance.
  vpc_security_group_ids = [aws_security_group.major2357_sg.id]
=======
resource "aws_instance" "major2_server" {
  ami           = "ami-0f918f7e67a3323f0"
  instance_type = "t2.large"
  key_name      = "major1" 
  
  # This attaches the security group defined above to the EC2 instance.
  vpc_security_group_ids = [aws_security_group.major2_sg.id]
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba

  # This script runs on the server's first boot to install Docker.
  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update -y
              sudo apt-get install -y docker.io
              sudo systemctl start docker
              sudo systemctl enable docker
              sudo usermod -aG docker ubuntu
              EOF
<<<<<<< HEAD
  iam_instance_profile = aws_iam_instance_profile.major2357_instance_profile.name
  tags = {
    Name = "major2357-Server-Terraform"
=======
  iam_instance_profile = aws_iam_instance_profile.major2_instance_profile.name
  root_block_device {
    volume_size = 20    # Size in GB
    volume_type = "gp3" # General Purpose SSD
  }
  tags = {
    Name = "major2-Server-Terraform"
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba
  }
}

# This resource runs a command locally on the GitHub runner AFTER the server is created.
# Its only job is to get the IP address and save it to a file for the next job to use.
resource "null_resource" "save_ip" {
  # This ensures the EC2 instance is fully created before this runs.
<<<<<<< HEAD
  depends_on = [aws_instance.major2357_server]
=======
  depends_on = [aws_instance.major2_server]
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba

  # This runs on the GitHub  runner itself.
  provisioner "local-exec" {
    # This command writes the clean IP address into a file named ip_address.txt
<<<<<<< HEAD
    command = "echo ${aws_instance.major2357_server.public_ip} > ip_address.txt"
=======
    command = "echo ${aws_instance.major2_server.public_ip} > ip_address.txt"
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba
  }
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.major2_server.public_ip
}
