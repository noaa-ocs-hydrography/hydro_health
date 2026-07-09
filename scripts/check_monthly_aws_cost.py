#!/usr/bin/env python3
import datetime
import subprocess
import re
from email.message import EmailMessage

# --- COST CONFIGURATION ---
EC2_RATES = {
    'i-06ebb5b58864f157f': 0.6120,  # c6a.4xlarge in us-east-1
    'i-0fc5fe193949551e5': 0.4637   # m7a.2xlarge in us-east-1
}
S3_BUCKET = "ocs-dev-csdl-hydrohealth"
S3_PRICE_PER_GB = 0.023  # us-east-2 standard tier

# --- EMAIL CONFIGURATION ---
# Replace these with your actual destination email addresses
SENDER_EMAIL = "stephen.patterson.lx@localhost"
RECIPIENT_EMAILS = ["stephen.patterson@noaa.gov", "aubrey.mccutchan@noaa.gov", "stephanie.watson@noaa.gov"]

def get_s3_bucket_size_gb(bucket_name):
    """Uses local AWS CLI to check bucket size."""
    try:
        result = subprocess.run(
            ['aws', 's3', 'ls', f's3://{bucket_name}', '--recursive', '--summarize'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return 0.0
        
        match = re.search(r"Total Size:\s+(\d+)", result.stdout)
        if match:
            total_bytes = int(match.group(1))
            return total_bytes / (1024 ** 3)
        return 0.0
    except Exception:
        return 0.0

def send_local_system_email(report_text, subject_line):
    """Pipes the email directly into the local Linux sendmail process without an SMTP password."""
    msg = EmailMessage()
    msg.set_content(report_text)
    msg['Subject'] = subject_line
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECIPIENT_EMAILS)
    
    try:
        # Calls the default Linux mail subsystem (/usr/sbin/sendmail)
        # -t reads the To/From/Subject from the message headers automatically
        p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE, text=True)
        p.communicate(msg.as_string())
        print("Email passed to local system mail queue successfully!")
    except Exception as e:
        print(f"Failed to hand off email to system mail queue: {e}")

def main():
    today = datetime.date.today()
    days_in_current_month = today.day
    
    first_of_this_month = today.replace(day=1)
    last_day_prev_month = first_of_this_month - datetime.timedelta(days=1)
    days_in_prev_month = last_day_prev_month.day

    # 1. Calculate EC2 Costs
    prev_ec2_total = sum(rate * 24 * days_in_prev_month for rate in EC2_RATES.values())
    curr_ec2_total = sum(rate * 24 * days_in_current_month for rate in EC2_RATES.values())

    # 2. Calculate S3 Costs
    print("Scanning S3 bucket size directly...")
    bucket_size_gb = get_s3_bucket_size_gb(S3_BUCKET)
    
    prev_s3_total = bucket_size_gb * S3_PRICE_PER_GB
    curr_s3_total = bucket_size_gb * S3_PRICE_PER_GB

    # Generate Report String
    report = []
    report.append("==================== AWS COST COMPARISON REPORT ====================")
    report.append(f"Previous Month Estimate ({last_day_prev_month.strftime('%B')}):")
    report.append(f"Current Month MTD ({today.strftime('%B')} 1st to {today.strftime('%d')})")
    report.append("--------------------------------------------------------------------")
    report.append(f"{'Resource Component':<30} | {'Prev Month':<12} | {'Current MTD':<12}")
    report.append("--------------------------------------------------------------------")
    report.append(f"{'EC2 Instances (Assumed 24/7)':<30} | ${prev_ec2_total:<11.2f} | ${curr_ec2_total:<11.2f}")
    report.append(f"{'S3 Bucket (Direct Scan)':<30} | ${prev_s3_total:<11.2f} | ${curr_s3_total:<11.2f}")
    report.append("--------------------------------------------------------------------")
    report.append(f"{'TOTAL ESTIMATED COST':<30} | ${prev_ec2_total + prev_s3_total:<11.2f} | ${curr_ec2_total + curr_s3_total:<11.2f}")
    report.append(f"Current Bucket Size: {bucket_size_gb:.2f} GB")
    report.append("====================================================================\n")
    
    report_text = "\n".join(report)
    
    # Print to stdout
    print(report_text)
    
    # Hand over to sendmail
    subject = f"AWS Cost Report: {today.strftime('%B %Y')} MTD Update"
    send_local_system_email(report_text, subject)

if __name__ == "__main__":
    main()