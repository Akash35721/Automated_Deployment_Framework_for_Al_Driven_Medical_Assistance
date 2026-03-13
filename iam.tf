# Creates an IAM role for the EC2 instance

resource "aws_iam_role" "major2357_ec2_role" {
  name = "major2357-EC2-Role1"


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

resource "aws_iam_instance_profile" "major2357_instance_profile" {
  name = "major2357-EC2-Instance-Profile1"
  role = aws_iam_role.major2357_ec2_role.name

}