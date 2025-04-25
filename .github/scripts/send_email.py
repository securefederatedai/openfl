import os
import smtplib
import logging
import argparse
from email.message import EmailMessage
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)

def send_email(sender_email: str, to_email: str, subject: str, email_body: str, smtp_user: str, smtp_pwd: str,
               smtp_email_server: str, cc_email: str = '', bcc_email: str = '', reply_email: str = '', is_html_body: bool = False,
               attachments: str = '') -> None:

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    if to_email:
        to_list = to_email.split(",")
        message["To"] = ", ".join(to_list)
    if cc_email:
        cc_list = cc_email.split(",")
        message["Cc"] = ", ".join(cc_list)
    if reply_email:
        message["Reply-To"] = reply_email
        sub_type = 'plain'
    if is_html_body:
        sub_type = 'html'
    message.set_content(email_body, subtype=sub_type)
    # Set up attachment if any
    if attachments:
        for attachment in attachments.split(','):
            with open(attachment, 'rb') as attachment_file:
                attachment_data = attachment_file.read()
            message.add_attachment(
                attachment_data,
                maintype='application',
                subtype='octet-stream',
                filename=os.path.basename(attachment)
            )
    logger.info(f'Setting smtp server {smtp_email_server}...')
    smtp_server = smtplib.SMTP(smtp_email_server)
    smtp_server.starttls()
    smtp_server.login(smtp_user, smtp_pwd)
    logger.info(f'smtp server authentication successful')
    try:
        logger.info(f'Sending email...')
        if bcc_email:
            # Send bcc list as an argument instead of adding it to the header to keep it hidden
            bcc_list = bcc_email.split(",")
            smtp_server.send_message(message, bcc=bcc_list)
        else:
            smtp_server.send_message(message)
        logger.info(f'email sent.')
    except Exception as ex:
        raise ex
    finally:
        try:
            smtp_server.quit()
        except smtplib.SMTPServerDisconnected:
            pass
        finally:
            logger.info("smtp connection is closed")

def main():
    parser = argparse.ArgumentParser(description="Send an email with optional attachments")
    parser.add_argument('--sender', required=True, help='Sender email address')
    parser.add_argument('--to', required=True, help='Recipient email address(es) (comma-separated)')
    parser.add_argument('--subject', required=True, help='Email subject')
    parser.add_argument('--body', required=True, help='Email body')
    parser.add_argument('--smtp-user', required=True, help='SMTP server username')
    parser.add_argument('--smtp-pwd', required=True, help='SMTP server password')
    parser.add_argument('--smtp-server', required=True, help='SMTP server address and port')
    parser.add_argument('--cc', default='', help='CC email address(es) (comma-separated)')
    parser.add_argument('--bcc', default='', help='BCC email address(es) (comma-separated)')
    parser.add_argument('--reply-to', default='', help='Reply-To email address')
    parser.add_argument('--html-body', action='store_true', help='Flag to indicate if email body is HTML')
    parser.add_argument('--attachments', default='', help='Attachment file path(s) (space-separated)')
    args = parser.parse_args()

    send_email(
        sender_email=args.sender,
        to_email=args.to,
        subject=args.subject,
        email_body=args.body,
        smtp_user=args.smtp_user,
        smtp_pwd=args.smtp_pwd,
        smtp_email_server=args.smtp_server,
        cc_email=args.cc,
        bcc_email=args.bcc,
        reply_email=args.reply_to,
        is_html_body=args.html_body,
        attachments=args.attachments
    )

if __name__ == '__main__':
    main()
