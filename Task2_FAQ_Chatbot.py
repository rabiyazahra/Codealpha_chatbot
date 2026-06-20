"""
FAQ Chatbot — CodeAlpha AI/ML Internship Task 2
Uses: sklearn (TF-IDF + Cosine Similarity), re (text preprocessing)
No external NLP library needed — pure Python + sklearn
"""

import re
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════
# STEP 1: FAQ DATASET (Food Delivery App — FoodRush)
# ══════════════════════════════════════════════════════════════
FAQ_DATA = [
    # Ordering
    {
        "question": "How do I place an order?",
        "answer": "To place an order: Open the app → Search for a restaurant → Select items → Add to cart → Proceed to checkout → Choose payment method → Place Order. You'll get a confirmation notification instantly!"
    },
    {
        "question": "Can I order from multiple restaurants at once?",
        "answer": "Sorry, each order can only be from one restaurant at a time. To order from multiple restaurants, you'll need to place separate orders."
    },
    {
        "question": "How do I track my order?",
        "answer": "After placing your order, go to 'My Orders' in the app. You'll see a live map with your delivery rider's location and estimated arrival time."
    },
    {
        "question": "Can I schedule an order for later?",
        "answer": "Yes! On the checkout page, tap 'Schedule Order' and pick your preferred date and time. We deliver up to 7 days in advance."
    },
    {
        "question": "How do I cancel my order?",
        "answer": "You can cancel your order within 2 minutes of placing it. Go to 'My Orders' → Select your order → Tap 'Cancel Order'. After 2 minutes, cancellation may not be possible."
    },
    # Payment
    {
        "question": "What payment methods are accepted?",
        "answer": "We accept: Credit/Debit Cards (Visa, Mastercard), JazzCash, EasyPaisa, Cash on Delivery, and FoodRush Wallet. All digital payments are 100% secure."
    },
    {
        "question": "How do I apply a promo code?",
        "answer": "On the checkout page, tap 'Apply Promo Code', enter your code, and hit Apply. The discount will be shown before you confirm your order."
    },
    {
        "question": "Why was my payment declined?",
        "answer": "Payment can be declined due to: insufficient balance, incorrect card details, expired card, or bank restrictions. Try a different payment method or contact your bank."
    },
    {
        "question": "Is my payment information safe?",
        "answer": "Absolutely! We use 256-bit SSL encryption and never store your full card details. All transactions are PCI-DSS compliant and fully secure."
    },
    {
        "question": "How do I get a refund?",
        "answer": "Refunds are processed within 3-5 business days to your original payment method. For wallet payments, the refund is instant. Contact support to initiate a refund."
    },
    # Delivery
    {
        "question": "How long does delivery take?",
        "answer": "Average delivery time is 20-35 minutes depending on your location and restaurant preparation time. The app shows the estimated time before you order."
    },
    {
        "question": "What is the delivery fee?",
        "answer": "Delivery fees range from free to Rs. 100 depending on distance and restaurant. Many restaurants offer free delivery on orders above Rs. 500."
    },
    {
        "question": "Do you deliver to my area?",
        "answer": "We currently deliver in Karachi, Lahore, and Islamabad. Enter your address on the home screen to see which restaurants are available near you."
    },
    {
        "question": "What if my order is late?",
        "answer": "If your order is significantly late, you can contact support via the app. We'll investigate and may offer a discount or credit for your inconvenience."
    },
    {
        "question": "Can I change my delivery address after ordering?",
        "answer": "You can change your delivery address only within 1 minute of placing the order. Go to 'My Orders' → Select order → 'Edit Address'."
    },
    # Account
    {
        "question": "How do I create an account?",
        "answer": "Download the FoodRush app → Tap 'Sign Up' → Enter your name, email, and phone number → Verify your phone via OTP → You're all set to order!"
    },
    {
        "question": "I forgot my password. How do I reset it?",
        "answer": "On the login screen, tap 'Forgot Password' → Enter your registered email → Check your email for a reset link → Set a new password. Done!"
    },
    {
        "question": "How do I update my profile information?",
        "answer": "Go to the Profile tab → Tap 'Edit Profile' → Update your name, email, phone, or profile picture → Tap 'Save Changes'."
    },
    {
        "question": "How do I delete my account?",
        "answer": "To delete your account, go to Profile → Settings → Account → 'Delete Account'. Please note this action is permanent and all order history will be lost."
    },
    {
        "question": "Can I have multiple delivery addresses?",
        "answer": "Yes! Go to Profile → 'Saved Addresses' → Tap '+' to add up to 5 addresses like Home, Work, and more. Select your preferred address at checkout."
    },
    # Food & Restaurants
    {
        "question": "How do I find vegetarian or vegan options?",
        "answer": "Use the 'Veg' filter on the Search screen to show only vegetarian restaurants and items. Many menus also have a 'V' badge next to vegetarian dishes."
    },
    {
        "question": "Can I customize my food order?",
        "answer": "Yes! Many restaurants allow customization like extra cheese, no onions, spice level, etc. Look for the 'Customize' option when adding an item to your cart."
    },
    {
        "question": "How are restaurants rated?",
        "answer": "Ratings are based on verified customer reviews after completed orders. Each restaurant displays its average rating out of 5 stars and total number of reviews."
    },
    {
        "question": "What if an item is unavailable?",
        "answer": "If an item you ordered becomes unavailable, the restaurant will contact you through the app to offer a substitute or remove it from your order with a refund."
    },
    # Support
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach us via: In-app Chat (fastest), Email: support@foodrush.pk, Phone: 0311-FOODRUSH (Mon-Sun 8am-11pm). Average response time is under 5 minutes!"
    },
    {
        "question": "What do I do if I received the wrong order?",
        "answer": "We're sorry! Go to 'My Orders' → Select the order → 'Report an Issue' → Select 'Wrong Order'. We'll arrange a replacement or full refund immediately."
    },
    {
        "question": "What if my food arrived cold or damaged?",
        "answer": "Report it via 'My Orders' → 'Report an Issue' → 'Food Quality'. Please include a photo. We'll investigate with the restaurant and offer you a refund or discount."
    },
    {
        "question": "How do I leave a review?",
        "answer": "After your order is delivered, you'll get a notification to rate your experience. Or go to 'My Orders' → Completed → 'Write a Review'. Your feedback helps us improve!"
    },
]

# ══════════════════════════════════════════════════════════════
# STEP 2: TEXT PREPROCESSING (NLP)
# ══════════════════════════════════════════════════════════════
STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or",
    "but","not","with","as","by","this","that","i","my","me","can","do",
    "how","what","when","where","why","which","who","will","be","was","are",
    "have","has","had","you","your","we","our","they","their","from","about"
}

def preprocess(text: str) -> str:
    """
    NLP Preprocessing Pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize (split)
    4. Remove stopwords
    5. Basic stemming (suffix stripping)
    6. Rejoin
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 3. Tokenize
    tokens = text.split()
    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # 5. Basic stemming (remove common suffixes)
    stemmed = []
    for t in tokens:
        if t.endswith("ing") and len(t) > 5:   t = t[:-3]
        elif t.endswith("tion") and len(t) > 5: t = t[:-4]
        elif t.endswith("ed") and len(t) > 4:   t = t[:-2]
        elif t.endswith("es") and len(t) > 3:   t = t[:-2]
        elif t.endswith("s") and len(t) > 3:    t = t[:-1]
        stemmed.append(t)
    # 6. Rejoin
    return " ".join(stemmed)


# ══════════════════════════════════════════════════════════════
# STEP 3: BUILD TF-IDF MODEL
# ══════════════════════════════════════════════════════════════
class FAQChatbot:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.questions = [item["question"] for item in faq_data]
        self.answers   = [item["answer"]   for item in faq_data]

        # Preprocess all questions
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Build TF-IDF vectorizer on processed questions
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

        # Greetings & fallbacks
        self.greetings = ["hi","hello","hey","salam","assalam","good morning","good evening","howdy"]
        self.thanks     = ["thanks","thank you","shukriya","jazakallah","thx"]
        self.bye        = ["bye","goodbye","exit","quit","khuda hafiz","alvida"]

        self.THRESHOLD = 0.15   # minimum cosine similarity to give an answer

    def get_response(self, user_input: str) -> dict:
        """
        Returns dict with:
          - answer: str
          - matched_question: str or None
          - confidence: float (0-1)
          - intent: str
        """
        raw = user_input.strip()
        lower = raw.lower()

        # Intent: greeting
        if any(g in lower for g in self.greetings):
            return {
                "answer": "Hello! Welcome to FoodRush Support. I'm here to help you with orders, payments, delivery, and more. What can I assist you with today?",
                "matched_question": None,
                "confidence": 1.0,
                "intent": "greeting"
            }

        # Intent: thanks
        if any(t in lower for t in self.thanks):
            return {
                "answer": "You're welcome! Is there anything else I can help you with?",
                "matched_question": None,
                "confidence": 1.0,
                "intent": "thanks"
            }

        # Intent: bye
        if any(b in lower for b in self.bye):
            return {
                "answer": "Goodbye! Thank you for using FoodRush. Have a delicious day!",
                "matched_question": None,
                "confidence": 1.0,
                "intent": "farewell"
            }

        # NLP matching via TF-IDF + Cosine Similarity
        processed_input = preprocess(raw)
        if not processed_input:
            return {
                "answer": "I didn't quite catch that. Could you please rephrase your question?",
                "matched_question": None,
                "confidence": 0.0,
                "intent": "unclear"
            }

        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix)[0]
        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score < self.THRESHOLD:
            return {
                "answer": "I'm sorry, I couldn't find a matching answer for your question. Please try rephrasing, or contact our support team at support@foodrush.pk or call 0311-FOODRUSH.",
                "matched_question": None,
                "confidence": best_score,
                "intent": "no_match"
            }

        return {
            "answer": self.answers[best_idx],
            "matched_question": self.questions[best_idx],
            "confidence": best_score,
            "intent": "faq_match"
        }


# ══════════════════════════════════════════════════════════════
# STEP 4: TERMINAL CHAT UI
# ══════════════════════════════════════════════════════════════
def run_chat():
    print("\n" + "="*60)
    print("   FoodRush FAQ Chatbot — CodeAlpha AI/ML Task 2")
    print("="*60)
    print("   NLP: TF-IDF + Cosine Similarity (sklearn)")
    print("   Preprocessing: Tokenize → Stopwords → Stemming")
    print("   FAQ Dataset: 28 questions across 6 categories")
    print("="*60)
    print("   Type your question | 'quit' to exit\n")

    bot = FAQChatbot(FAQ_DATA)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Have a great day!")
            break

        result = bot.get_response(user_input)

        print(f"\nBot: {result['answer']}")
        if result['matched_question']:
            conf_pct = result['confidence'] * 100
            print(f"     [Matched: \"{result['matched_question']}\" | Confidence: {conf_pct:.1f}%]")
        print()


if __name__ == "__main__":
    run_chat()
