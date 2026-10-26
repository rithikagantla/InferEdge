"""Held-out labeled prompts for measuring semantic-router accuracy.

None of these appear in the router's category exemplars
(app/routing/semantic_router.py) or in the benchmark prompts
(app/sample_prompts.py). Many are phrased without the obvious keyword
("refund", "password", "package") so a router has to match meaning, not
words.
"""
from typing import List, Tuple

ROUTER_EVAL_SET: List[Tuple[str, str]] = [
    # billing
    ("There's a line on my statement from you that I don't recognize.", "billing"),
    ("You took money from my account on the 3rd and again on the 4th.", "billing"),
    ("Why does my receipt show tax when it didn't last month?", "billing"),
    ("Can I switch to paying yearly instead of monthly?", "billing"),
    ("My company needs a VAT number added to past invoices.", "billing"),
    ("The promo code I used didn't reduce what I paid.", "billing"),
    ("I got billed for a seat we removed weeks ago.", "billing"),
    ("How do I pay with PayPal instead of a card?", "billing"),
    ("My bank flagged your charge as suspicious, is it legit?", "billing"),
    ("What day of the month will I be charged going forward?", "billing"),
    # technical_support
    ("Nothing loads past the spinning wheel on the dashboard.", "technical_support"),
    ("Exports to CSV come out with garbled characters.", "technical_support"),
    ("The webhook stopped firing after we rotated our keys.", "technical_support"),
    ("Notifications aren't showing up on my Android phone.", "technical_support"),
    ("Search returns nothing even for items I know exist.", "technical_support"),
    ("I keep getting logged out every few minutes on Safari.", "technical_support"),
    ("The SDK throws a null pointer exception on startup.", "technical_support"),
    ("Images upload but show as broken thumbnails.", "technical_support"),
    ("Dark mode makes half the buttons invisible.", "technical_support"),
    ("Sync conflicts are duplicating all my calendar events.", "technical_support"),
    # account_management
    ("I can't get into my profile anymore, the login link expired.", "account_management"),
    ("How do I transfer ownership of the workspace to my colleague?", "account_management"),
    ("Please delete my account and all my data.", "account_management"),
    ("I want to change my username.", "account_management"),
    ("Can I merge two accounts I accidentally created?", "account_management"),
    ("My authenticator app was on my old phone and now I'm locked out.", "account_management"),
    ("How do I remove a former employee's access?", "account_management"),
    ("I'd like to move from the basic tier to the pro tier.", "account_management"),
    ("Where do I update my phone number for security codes?", "account_management"),
    ("Can I pause my membership for two months?", "account_management"),
    # refund
    ("The jacket didn't fit, can I get my money back?", "refund"),
    ("I cancelled within the trial but was still charged, please reverse it.", "refund"),
    ("You said I'd be reimbursed for the late delivery, when?", "refund"),
    ("I'd like a credit back for the hours your service was down.", "refund"),
    ("Return label printed, when does the money come back?", "refund"),
    ("The concert was cancelled, how do I get reimbursed for tickets?", "refund"),
    ("Can I get compensated for the broken item instead of a replacement?", "refund"),
    ("I was promised a price adjustment after the sale started.", "refund"),
    ("The course was nothing like described, I want my payment returned.", "refund"),
    ("Is the restocking fee taken out of my reimbursement?", "refund"),
    # shipping
    ("The courier says delivered but nothing is at my door.", "shipping"),
    ("Can you send it to my office instead of my home?", "shipping"),
    ("How much does express delivery to Canada cost?", "shipping"),
    ("My parcel has been stuck in customs for a week.", "shipping"),
    ("Only two of the three boxes arrived.", "shipping"),
    ("Can I pick the order up from a locker?", "shipping"),
    ("The box was soaked and the contents are ruined.", "shipping"),
    ("When will the backordered item actually ship?", "shipping"),
    ("Do you deliver on weekends?", "shipping"),
    ("I need it before Friday, can you upgrade the shipping?", "shipping"),
    # human_escalation
    ("I've explained this four times, get me someone who can actually decide.", "human_escalation"),
    ("Transfer me to a supervisor please.", "human_escalation"),
    ("I'm not talking to a chatbot about this, put a person on.", "human_escalation"),
    ("I'm considering legal action unless someone senior calls me.", "human_escalation"),
    ("Can I schedule a phone call with your support team?", "human_escalation"),
    ("This is unacceptable and I want it reviewed by management.", "human_escalation"),
    ("Escalate my case, the ticket number is 55120.", "human_escalation"),
    ("Is there an actual human I can email directly?", "human_escalation"),
    ("I'd like to speak to whoever handles complaints.", "human_escalation"),
    ("Stop sending me canned answers and connect me to staff.", "human_escalation"),
]
