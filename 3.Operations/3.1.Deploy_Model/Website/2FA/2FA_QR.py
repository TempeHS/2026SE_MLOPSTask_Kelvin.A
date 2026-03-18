import pyotp
import qrcode


def gen_key():
    return pyotp.random_base32()


def gen_url(key):
    return pyotp.totp.TOTP(key).provisioning_uri(name="bob", issuer_name="2FA App")


def verify_code(key: str, code: str):
    totp = pyotp.TOTP(key)
    return totp.verify(code)


if __name__ == "__main__":
    # Generate key and QR code
    user_key = gen_key()
    user_totp = pyotp.TOTP(user_key)
    qr_uri = gen_url(user_key)

    qrcode.make(qr_uri).save("NCode.png")

    while True:
        user_code = input("enter code: ")
        print(verify_code(user_key, user_code))
