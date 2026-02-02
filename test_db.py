import db_manager as db

print("Testing database...")
try:
    result = db.add_user('test@example.com', 'Test User', 'testpass123')
    print(f'Add user result: {result}')

    verify = db.verify_user('test@example.com', 'testpass123')
    print(f'Verify user result: {verify}')

    user = db.get_user('test@example.com')
    print(f'Get user result: {user}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
