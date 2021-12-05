from app.mail import mail
from app import app

def send_mail_active_user(active_code):
    code= active_code['code']
    email= active_code['email']
    link_active = app.config['SERVER_ADDRESS'] + f'/api/user/active/?email={email}&code={code}'
    content = 'Vui lòng nhấn vào link để kích hoạt tài khoản: ' + link_active
    mail.send_mail_without_template(email, '[Fresh] Kích hoạt tài khoản', content=content)

    return True

def send_mail_reset_password(email,password):
    try:
        # link_reset = app.config['SERVER_ADDRESS'] + f'/api/user/reset/?email={reset_code.email}&code={reset_code.code}'
        content = 'Your new password: ' + password
        mail.send_mail_without_template(email, '[Fresh] Reset mật khẩu', content=content)
    except:
        return False
    return True