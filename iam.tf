# Creates an IAM role for the EC2 instance
<<<<<<< HEAD
resource "aws_iam_role" "major2357_ec2_role" {
  name = "major2357-EC2-Role1"
=======
resource "aws_iam_role" "major2_ec2_role" {
  name = "major2-EC2-Role"
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba

  # The policy that allows EC2 to assume this role
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}
# Creates an instance profile that can be associated with the EC2 instance
<<<<<<< HEAD
resource "aws_iam_instance_profile" "major2357_instance_profile" {
  name = "major2357-EC2-Instance-Profile1"
  role = aws_iam_role.major2357_ec2_role.name
=======
resource "aws_iam_instance_profile" "major2_instance_profile" {
  name = "major2-EC2-Instance-Profile"
  role = aws_iam_role.major2_ec2_role.name
>>>>>>> 22731154287d3879e31b0a411e6d6bcbabdc19ba
}