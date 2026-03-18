import pyotp
import time


def gen_key():
    return pyotp.random_base32()


def gen_url(key):
    return pyotp.totp.TOTP(key).provisioning_uri(name="bob", issuer_name="2fa App")


def generate_code(key: str):
    totp = pyotp.TOTP(key)
    return totp.now()


def verify_code(key: str, code: str):
    totp = pyotp.TOTP(key)
    return totp.verify(code)


if __name__ == "__main__":
    # TODO: Put in Database
    user_key = gen_key()

    print(user_key)

    qr_uri = gen_url(user_key)

    print(qr_uri)

    user_code = generate_code(user_key)
    print(user_code)

    time.sleep(30)
    user_code2 = generate_code(user_key)

    print(verify_code(user_key, user_code))
    print(verify_code(user_key, user_code2))
