"""Synthetic email dataset generator for testing and training."""
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

DOMAINS_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "tech_support": [
        {
            "subject": "Cannot log in to my account",
            "body": "Hi, I have been trying to log in to my account for the past hour but I keep getting an 'Invalid credentials' error. My username is correct and I've reset my password twice. Please help.",
            "intent": "complaint",
            "reply": "Dear Customer,\n\nThank you for contacting us. We apologize for the inconvenience. Your account may have been temporarily locked due to multiple failed attempts. We have unlocked your account. Please try again and contact us if the issue persists.\n\nBest regards,\nTech Support Team",
        },
        {
            "subject": "How to integrate the API?",
            "body": "Hello, I am a developer trying to integrate your REST API into our application. Could you please provide documentation or examples for authentication and the /users endpoint?",
            "intent": "inquiry",
            "reply": "Dear Developer,\n\nThank you for your interest in our API. You can find our full documentation at https://docs.example.com/api. For authentication, we use OAuth 2.0 Bearer tokens. Please see the /users endpoint guide in Section 3.\n\nBest regards,\nDeveloper Support",
        },
        {
            "subject": "Software crashes on startup",
            "body": "The application crashes immediately when I try to start it. I am running Windows 11 with 16GB RAM. Error code: 0xC000007B.",
            "intent": "complaint",
            "reply": "Dear Customer,\n\nThank you for reporting this. Error 0xC000007B typically indicates a missing or corrupt DLL. Please try reinstalling the Visual C++ Redistributable package from Microsoft's website. If the issue persists, please send us the crash log.\n\nBest regards,\nSupport Team",
        },
    ],
    "sales": [
        {
            "subject": "Pricing inquiry for Enterprise plan",
            "body": "Hi, we are a company of 500 employees looking to subscribe to your Enterprise plan. Could you provide pricing details and volume discounts?",
            "intent": "inquiry",
            "reply": "Dear Valued Prospect,\n\nThank you for your interest in our Enterprise plan. For 500 users, we offer significant volume discounts. Our Enterprise pricing starts at $15/user/month with annual billing. I'd be happy to arrange a call to discuss your specific needs and provide a custom quote.\n\nBest regards,\nSales Team",
        },
        {
            "subject": "Request for product demo",
            "body": "Hello, I am the IT manager at ABC Corp and I would like to schedule a demo of your project management software. We are currently evaluating solutions for our 200-person team.",
            "intent": "scheduling",
            "reply": "Dear IT Manager,\n\nThank you for reaching out! We'd be delighted to give you a personalized demo. I have openings on Tuesday and Thursday this week at 2 PM or 4 PM EST. Please let me know your preferred time and I'll send a calendar invite.\n\nBest regards,\nSales Representative",
        },
    ],
    "hr": [
        {
            "subject": "Vacation request for next month",
            "body": "Hi HR, I would like to request 5 days of annual leave from March 15-19. I have 12 days remaining. My manager has verbally approved this. Please confirm.",
            "intent": "request",
            "reply": "Dear Employee,\n\nThank you for your leave request. We have received and approved your request for 5 days annual leave from March 15-19. Your leave balance will be updated to 7 days. Please ensure your work is covered during your absence.\n\nBest regards,\nHR Department",
        },
        {
            "subject": "Question about parental leave policy",
            "body": "Hello, I recently found out I'm expecting and I want to understand our company's parental leave policy. How many weeks are available and when should I notify HR?",
            "intent": "inquiry",
            "reply": "Dear Employee,\n\nCongratulations on your wonderful news! Our parental leave policy provides 12 weeks of fully paid leave for primary caregivers and 4 weeks for secondary caregivers. Please notify HR at least 30 days before your expected start date. We have a full guide available on the HR portal.\n\nBest regards,\nHR Department",
        },
    ],
    "project_management": [
        {
            "subject": "Project status update needed",
            "body": "Hi team, the quarterly review is next Friday and I need status updates on all active projects. Please send a brief summary of progress, blockers, and next steps for your respective projects by EOD Wednesday.",
            "intent": "request",
            "reply": "Dear Project Lead,\n\nThank you for the reminder. I will have the status update for Project Alpha ready by EOD Wednesday. Current progress: 70% complete. Main blocker: awaiting vendor approval on third-party integration. Next steps: Complete API testing, final QA.\n\nBest regards,\nProject Manager",
        },
        {
            "subject": "Deadline extension request",
            "body": "Dear Manager, due to unexpected technical challenges with the database migration, our team requires a 2-week extension on the Q1 deliverable. The original deadline is March 31. Can we move it to April 14?",
            "intent": "request",
            "reply": "Dear Team Lead,\n\nThank you for the advance notice. I understand database migrations can present unexpected challenges. I am approving a one-week extension to April 7. Please provide a detailed mitigation plan and daily progress updates starting Monday.\n\nBest regards,\nProject Director",
        },
    ],
    "customer_service": [
        {
            "subject": "Order not received - Order #12345",
            "body": "I placed an order 10 days ago (Order #12345) and it still hasn't arrived. The tracking shows it's been 'in transit' for 5 days. I need this by Friday for an important event. Please help!",
            "intent": "complaint",
            "reply": "Dear Customer,\n\nThank you for contacting us about Order #12345. I sincerely apologize for the delay. I have escalated this to our logistics team and we are investigating the shipment status. I will provide an update within 24 hours. If it cannot arrive by Friday, we will arrange expedited re-shipment at no cost.\n\nBest regards,\nCustomer Service Team",
        },
        {
            "subject": "Refund request for defective product",
            "body": "Hello, I purchased the XR-5000 headphones last week (Invoice #INV-8901) but one ear cup stopped working after 2 days. I would like a full refund or replacement under your warranty policy.",
            "intent": "complaint",
            "reply": "Dear Customer,\n\nThank you for reporting this. I'm sorry to hear about the defective headphones. As this is within our 30-day return policy and covered under warranty, I am processing a full replacement for you. You will receive a prepaid return label via email within 2 hours. Your replacement will ship within 1-2 business days upon receipt.\n\nBest regards,\nCustomer Service",
        },
        {
            "subject": "Follow up on open support ticket #45678",
            "body": "Hi, I opened a support ticket (#45678) two weeks ago regarding billing issues and haven't received a resolution yet. Can you please provide an update?",
            "intent": "follow_up",
            "reply": "Dear Customer,\n\nThank you for following up on ticket #45678. I apologize for the extended wait. I have reviewed your case and our billing team has identified the issue. The overcharge of $49.99 will be refunded to your original payment method within 5-7 business days. You will receive a confirmation email shortly.\n\nBest regards,\nSupport Team",
        },
        {
            "subject": "Thank you for excellent service",
            "body": "Hi, I just wanted to say thank you to your support team, especially Sarah, who helped me resolve a complex billing issue last week. She was incredibly patient and professional. You have a customer for life!",
            "intent": "thank_you",
            "reply": "Dear Valued Customer,\n\nThank you so much for your wonderful feedback! We are thrilled to hear about your positive experience with Sarah. We will certainly share your kind words with her and the team. It is always a pleasure to serve customers like you.\n\nBest regards,\nCustomer Service Team",
        },
    ],
}

SENDERS = [
    "john.doe@company.com", "jane.smith@corp.org", "michael.brown@enterprise.net",
    "sarah.johnson@business.com", "david.wilson@startup.io", "emily.davis@firm.co",
    "robert.taylor@agency.com", "lisa.anderson@client.org",
]

RECIPIENTS = [
    "support@example.com", "hr@example.com", "sales@example.com",
    "info@example.com", "team@example.com",
]


class SyntheticEmailGenerator:
    """Generates synthetic email datasets for testing and evaluation."""

    def generate_dataset(
        self,
        count: int = 30,
        domains: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a synthetic email dataset.

        Args:
            count: Number of emails to generate.
            domains: List of domains to include. Defaults to all domains.

        Returns:
            List of synthetic email dictionaries.
        """
        if domains is None:
            domains = list(DOMAINS_TEMPLATES.keys())

        emails = []
        base_date = datetime(2025, 1, 1)

        # Collect all templates from selected domains
        templates = []
        for domain in domains:
            for template in DOMAINS_TEMPLATES.get(domain, []):
                templates.append((domain, template))

        if not templates:
            logger.warning("No templates found for domains: %s", domains)
            return []

        for i in range(count):
            domain, template = templates[i % len(templates)]
            sender = random.choice(SENDERS)
            recipient = random.choice(RECIPIENTS)
            date = base_date + timedelta(days=i * 3, hours=random.randint(8, 17))

            email = {
                "id": f"email_{i + 1:03d}",
                "subject": template["subject"],
                "body": template["body"],
                "sender": sender,
                "recipient": recipient,
                "date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "intent": template["intent"],
                "reply": template["reply"],
                "domain": domain,
            }
            emails.append(email)

        logger.info("Generated %d synthetic emails across %d domains", len(emails), len(domains))
        return emails


