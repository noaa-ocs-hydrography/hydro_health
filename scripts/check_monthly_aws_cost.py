#!/usr/bin/env python3
import calendar 
import datetime
import subprocess
import os
import re
import sys
from email.message import EmailMessage

"""
SETUP
1. Install mail server if using a new EC2 - sudo dnf install postfix
2. sudo postfix check
3. sudo systemctl enable postfix
4. sudo systemctl start postfix
5. systemctl is-enabled postfix   # check if is-enabled returns enabled for automation
6. sudo dnf install -y cronie  # install crontab
7. sudo systemctl enable --now crond  # enable and start the crontab
8. sudo journalctl -u crond -b --no-pager  # see what cron has run some last boot
9. sudo journalctl -u crond -b | grep -i check_monthly_aws_cost  - grep if cron ran the task
10. --force  # flag to run script for testing

- crontab contents with required blank space after:
PATH=/usr/local/bin:/usr/bin:/bin

@reboot sleep 120 && /home/stephen.patterson.lx/miniconda3/envs/hydro_health/bin/python /home/stephen.patterson.lx/Repos/hydro_health/scripts/check_monthly_aws_cost.py > /dev/null 2>&1

"""


# ==============================================================================
# COST CONFIGURATION
# ==============================================================================
# Standard hourly pricing (us-east-1)
HOURLY_PUBLIC_IPV4_RATE = 0.005  # $0.005/hr per in-use or idle public IPv4
EBS_GP3_PER_GB_MONTH = 0.08      # ~$0.08/GB-month for general purpose SSD
S3_PRICE_PER_GB = 0.023          # us-east-2 standard tier ($/GB-month)

S3_BUCKET = "ocs-dev-csdl-hydrohealth"

# State file to track execution once per calendar week
TRACK_FILE = "/home/stephen.patterson.lx/.last_cost_report_week"

# Define instances with their attached fixed infrastructure:
INSTANCES = {
    'i-0c7fdea6096d85b56': {
        'name': 'r8a.4xlarge',
        'compute_hourly': 1.2781,
        'ebs_size_gb': 1000,      # Adjust to your root + data volume size
        'has_public_ip': True    # True adds ~$3.65/month IPv4 fee
    },
    'i-089b1b07a7162bea7': {
        'name': 'm7a.2xlarge',
        'compute_hourly': 0.4637,
        'ebs_size_gb': 1000,      # Adjust to your root + data volume size
        'has_public_ip': True
    }
}

# --- EMAIL CONFIGURATION ---
SENDER_EMAIL = "stephen.patterson.lx@localhost"
RECIPIENT_EMAILS = ["stephen.patterson@noaa.gov", "aubrey.mccutchan@noaa.gov", "stephanie.watson@noaa.gov"]
# RECIPIENT_EMAILS = ["stephen.patterson@noaa.gov"]


def should_run_this_week():
    """Returns True if the script has not yet run during this ISO calendar week."""
    # Allow manual override via command line: python script.py --force
    if "--force" in sys.argv:
        return True

    today = datetime.date.today()
    iso_year, iso_week, _ = today.isocalendar()
    current_week_str = f"{iso_year}-W{iso_week:02d}"

    if os.path.exists(TRACK_FILE):
        try:
            with open(TRACK_FILE, "r") as f:
                recorded_week = f.read().strip()
            if recorded_week == current_week_str:
                print(f"Report already dispatched for week {current_week_str}. Exiting.")
                return False
        except Exception as e:
            print(f"Warning: Could not read week tracking file: {e}")

    return True


def mark_week_as_completed():
    """Writes the current ISO week to the tracker file."""
    today = datetime.date.today()
    iso_year, iso_week, _ = today.isocalendar()
    current_week_str = f"{iso_year}-W{iso_week:02d}"

    try:
        with open(TRACK_FILE, "w") as f:
            f.write(current_week_str)
        # Ensure both root and stephen.patterson.lx can overwrite the file
        os.chmod(TRACK_FILE, 0o666)
    except Exception as e:
        print(f"Warning: Could not save week tracking file: {e}")

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
    msg = EmailMessage()
    msg.set_content(report_text)
    msg['Subject'] = subject_line
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECIPIENT_EMAILS)
    
    try:
        p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE, text=True)
        p.communicate(msg.as_string())
        print("Email queued successfully.")
    except Exception as e:
        print(f"Failed to hand off email: {e}")


def main():
    if not should_run_this_week():
        return

    today = datetime.date.today()
    days_in_current_month = today.day
    
    first_of_this_month = today.replace(day=1)
    last_day_prev_month = first_of_this_month - datetime.timedelta(days=1)
    days_in_prev_month = last_day_prev_month.day

    _, total_days_in_current_month = calendar.monthrange(today.year, today.month)
    mtd_fraction = days_in_current_month / total_days_in_current_month

    # 1. Compute & IP Costs (Hourly)
    hourly_compute_total = sum(i['compute_hourly'] for i in INSTANCES.values())
    hourly_ip_total = sum(HOURLY_PUBLIC_IPV4_RATE for i in INSTANCES.values() if i['has_public_ip'])
    
    prev_compute_cost = hourly_compute_total * 24 * days_in_prev_month
    curr_compute_cost = hourly_compute_total * 24 * days_in_current_month

    prev_ip_cost = hourly_ip_total * 24 * days_in_prev_month
    curr_ip_cost = hourly_ip_total * 24 * days_in_current_month

    # 2. EBS Volume Storage Costs (Monthly GB-month)
    total_ebs_gb = sum(i['ebs_size_gb'] for i in INSTANCES.values())
    prev_ebs_cost = total_ebs_gb * EBS_GP3_PER_GB_MONTH
    curr_ebs_cost = total_ebs_gb * EBS_GP3_PER_GB_MONTH * mtd_fraction

    # 3. S3 Costs (Direct Scan)
    print("Scanning S3 bucket size...")
    bucket_size_gb = get_s3_bucket_size_gb(S3_BUCKET)
    
    prev_s3_cost = bucket_size_gb * S3_PRICE_PER_GB
    curr_s3_cost = bucket_size_gb * S3_PRICE_PER_GB * mtd_fraction

    # Totals
    prev_total = prev_compute_cost + prev_ip_cost + prev_ebs_cost + prev_s3_cost
    curr_total = curr_compute_cost + curr_ip_cost + curr_ebs_cost + curr_s3_cost

    # 4. Generate Report
    report = [
        "==================== AWS COST ESTIMATE REPORT ====================",
        f"Previous Month ({last_day_prev_month.strftime('%B %Y')} - {days_in_prev_month} days):",
        f"Current Month MTD ({today.strftime('%B 1')} to {today.strftime('%d, %Y')}):",
        "-" * 70,
        f"{'Cost Component':<34} | {'Prev Month':<12} | {'Current MTD':<12}",
        "-" * 70,
        f"{'EC2 Compute (24/7)':<34} | ${prev_compute_cost:<11.2f} | ${curr_compute_cost:<11.2f}",
        f"{'Public IPv4 Addresses':<34} | ${prev_ip_cost:<11.2f} | ${curr_ip_cost:<11.2f}",
        f"{f'EBS Storage ({total_ebs_gb} GB gp3)':<34} | ${prev_ebs_cost:<11.2f} | ${curr_ebs_cost:<11.2f}",
        f"{f'S3 Storage ({bucket_size_gb:.1f} GB)':<34} | ${prev_s3_cost:<11.2f} | ${curr_s3_cost:<11.2f}",
        "-" * 70,
        f"{'TOTAL ESTIMATED COST':<34} | ${prev_total:<11.2f} | ${curr_total:<11.2f}",
        "==================================================================",
        "* Note: Excludes variable cross-region/internet data transfer fees.",
        ""
    ]
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 5. Hand over to sendmail and record successful run
    subject = f"AWS Cost Estimate: {today.strftime('%B %Y')} Weekly Update"
    send_local_system_email(report_text, subject)
    mark_week_as_completed()


if __name__ == "__main__":
    main()