import sqlite3 as sql
import bcrypt
import pyotp
import qrcode
import io
import base64

DB = "databaseFiles/database.db"


def getUsers():
    con = sql.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT * FROM id7_tusers")
    con.commit()
    con.close()


def init_2fa_column():
    """Add twofa_key column if it doesn't exist"""
    try:
        con = sql.connect(DB)
        cur = con.cursor()
        cur.execute("ALTER TABLE id7_tusers ADD COLUMN twofa_key TEXT DEFAULT NULL")
        con.commit()
        con.close()
    except sql.OperationalError:
        # Column already exists
        pass


def init_dev_logs_table():
    """Create developer logs table if it doesn't exist"""
    con = sql.connect(DB)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dev_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            developer TEXT NOT NULL,
            time_worked REAL NOT NULL,
            repo_link TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    con.commit()
    con.close()


def add_dev_log(message, developer, time_worked, repo_link):
    """Add a new developer log entry"""
    if not message or not developer or time_worked is None or not repo_link:
        return False

    con = sql.connect(DB)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO dev_logs (message, developer, time_worked, repo_link) VALUES (?, ?, ?, ?)",
        (message, developer, time_worked, repo_link),
    )
    con.commit()
    con.close()
    return True


def get_dev_logs(sort_by="newest"):
    """Retrieve all developer logs, sorting options"""
    con = sql.connect(DB)
    cur = con.cursor()

    if sort_by == "oldest":
        order_clause = "ORDER BY timestamp ASC"
    elif sort_by == "developer":
        order_clause = "ORDER BY developer ASC, timestamp DESC"
    else:
        order_clause = "ORDER BY timestamp DESC"

    cur.execute(
        f"SELECT id, message, developer, time_worked, repo_link, timestamp FROM dev_logs {order_clause}"
    )
    logs = cur.fetchall()
    con.close()
    return logs


def NewUser(email, password):
    if not email or not password:
        return (False, "Email and password required")

    con = sql.connect(DB)
    cur = con.cursor()
    try:
        # Hash the password before storing
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cur.execute(
            "INSERT INTO id7_tusers (username, password) VALUES (?,?)",
            (email, hashed),
        )
        con.commit()
        con.close()
        return (True, "User created")
    except sql.IntegrityError:
        con.close()
        return (False, "User already exists")
    except sql.Error as e:
        try:
            con.close()
        except sql.Error:
            pass
        return (False, str(e))


def authenticate(email, password):
    if not email or not password:
        return False

    try:
        con = sql.connect(DB)
        cur = con.cursor()
        cur.execute("SELECT password FROM id7_tusers WHERE username =?", (email,))
        row = cur.fetchone()
        con.close()

        if not row:
            return False

        stored = row[0]

        if isinstance(stored, (bytes, bytearray, memoryview)):
            stored_bytes = bytes(stored)
            return bcrypt.checkpw(password.encode("utf-8"), stored_bytes)
        else:
            return password == stored
    except sql.Error:
        return False


def get_2fa_key(email):
    """Get or create 2FA key for a user"""
    try:
        con = sql.connect(DB)
        cur = con.cursor()
        cur.execute("SELECT twofa_key FROM id7_tusers WHERE username =?", (email,))
        row = cur.fetchone()

        if row and row[0]:
            con.close()
            return row[0]
        else:
            # Generate new key
            new_key = pyotp.random_base32()
            cur.execute(
                "UPDATE id7_tusers SET twofa_key =? WHERE username =?", (new_key, email)
            )
            con.commit()
            con.close()
            return new_key
    except sql.Error:
        try:
            con.close()
        except sql.Error:
            pass
        return None


def verify_2fa_code(email, code):
    """Verify 2FA code for a user"""
    try:
        key = get_2fa_key(email)
        if not key:
            return False

        totp = pyotp.TOTP(key)
        return totp.verify(code, valid_window=1)
    except (ValueError, TypeError):
        return False


def get_2fa_qr_uri(email):
    """Get QR code URI for 2FA setup"""
    try:
        key = get_2fa_key(email)
        if not key:
            return None

        totp = pyotp.TOTP(key)
        return totp.provisioning_uri(name=email, issuer_name="Secure Flask PWA")
    except (ValueError, TypeError):
        return None


def get_2fa_qr_code_base64(email):
    """Generate QR code as base64 data URI"""
    try:
        uri = get_2fa_qr_uri(email)
        if not uri:
            return None

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)

        # Create image
        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_base64}"
    except (ValueError, TypeError, OSError):
        return None
