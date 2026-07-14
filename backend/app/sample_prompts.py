"""Realistic customer-support prompts used for demos and benchmarks."""
from typing import List

SAMPLE_PROMPTS: List[str] = [
    # shipping
    "My order is delayed. Can you help?",
    "Where is my package? Tracking has not updated in 4 days.",
    "Can I change the delivery address on order #48213?",
    "Do you ship internationally, and how long does it take?",
    "My package arrived damaged. What should I do?",
    # billing
    "I was charged twice this month.",
    "Why did my subscription price go up?",
    "Can you send me an invoice for my last payment?",
    "My credit card was declined but I was still charged.",
    "How do I update my billing information?",
    # account management
    "I forgot my password and cannot log in.",
    "Can I upgrade my subscription?",
    "How do I change the email on my account?",
    "I want to add a second user to my team plan.",
    "How do I enable two-factor authentication?",
    # refunds
    "I want a refund for my last order.",
    "How long does a refund take to show up on my card?",
    "I returned the item two weeks ago and still have no refund.",
    "Can I get a partial refund for the damaged part of my order?",
    "What is your refund policy for digital purchases?",
    # technical support
    "The app keeps crashing after the latest update.",
    "I'm getting a 500 error when I try to check out.",
    "The mobile app won't sync with my desktop account.",
    "Video playback is really laggy on my smart TV app.",
    "Your API returns a timeout when I upload files over 10MB.",
    # human escalation
    "This is the third time I'm contacting you and nothing is fixed. Let me talk to a person.",
    "I need to speak with a human agent right now.",
    "Your bot answers are useless, connect me to a real support rep.",
    "I want to file a formal complaint with a manager.",
    "Please escalate this ticket, it has been open for two weeks.",
]
