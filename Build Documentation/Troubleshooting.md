# Troubleshooting Guide

**Version:** 1.1  
**Last Updated:** August 13, 2025  
**Sprint:** 2 - Tool Registry Service & Merchant Dashboard  
**Ticket:** TICKET-2012  

## Overview

This guide provides solutions to common issues encountered when working with the Agent Tool Marketplace authentication system. All issues listed here were encountered and resolved during development.

---

## Supabase Connection Issues

### ❌ "Certificate has expired" Error

**Problem:**
```
TypeError: fetch failed
  cause: Error: certificate has expired
    code: 'CERT_HAS_EXPIRED'
```

**Root Cause:**
Using incorrect URL format for Supabase API calls. The database hostname cannot be used directly for API requests.

**Solution:**
✅ **Fix:** Use correct URL conversion in `supabaseConfig.js`

```javascript
// ❌ WRONG - Using database hostname directly
const supabaseUrl = `https://${process.env.DB_HOST}`;
// Results in: https://db.project-id.supabase.co (invalid for API)

// ✅ CORRECT - Convert to API hostname  
const apiHost = process.env.DB_HOST?.replace(/^db\./, '') || process.env.DB_HOST;
const supabaseUrl = `https://${apiHost}`;
// Results in: https://project-id.supabase.co (valid for API)
```

**Verification:**
```bash
# Test the corrected URL
curl -I https://zpwvrbabgegtyfnwdfjb.supabase.co
# Should return HTTP/2 200 or 404 (not certificate error)
```

### ❌ DNS Resolution Failures

**Problem:**
```
Error: getaddrinfo ENOTFOUND db.project-id.supabase.co
```

**Root Cause:**
The `DB_HOST` contains the database connection URL, but API calls need the project API URL.

**Solution:**
✅ **Verify URL conversion is working:**

```bash
# Check current environment variables
node testEnvVariables.js

# Verify URL conversion
node -e "
const host = 'db.zpwvrbabgegtyfnwdfjb.supabase.co';
const apiHost = host.replace(/^db\./, '');
console.log('Database URL:', host);
console.log('API URL:', apiHost);
"
```

---

## Environment Variable Issues

### ❌ "Missing Supabase configuration" Error

**Problem:**
```
Error: Missing Supabase configuration. Please set DB_HOST and DB_ANON_KEY environment variables.
```

**Root Cause:**
Environment variables not being loaded correctly or missing from `.env` file.

**Diagnosis:**
```bash
# Test environment variable loading
node testEnvVariables.js
```

**Solution:**
✅ **Verify .env file location and content:**

1. **Check .env file exists:**
   ```bash
   ls -la .env
   # Should show file with size > 0 bytes
   ```

2. **Verify .env content:**
   ```bash
   # Check required variables exist (without showing values)
   grep -E "^(DB_HOST|DB_ANON_KEY)=" .env
   ```

3. **Test dotenv loading:**
   ```javascript
   require('dotenv').config();
   console.log('DB_HOST exists:', !!process.env.DB_HOST);
   console.log('DB_ANON_KEY exists:', !!process.env.DB_ANON_KEY);
   ```

### ❌ Environment Variables Not Loading from Subdirectories

**Problem:**
Environment variables work from project root but not from subdirectories.

**Root Cause:**
`dotenv` looks for `.env` file in current working directory, not relative to script location.

**Solution:**
✅ **Always run scripts from project root:**

```bash
# ❌ WRONG - Running from subdirectory
cd services/user-authentication-service
node testSupabase.js

# ✅ CORRECT - Running from project root
cd /path/to/project-root
node services/user-authentication-service/testSupabase.js
```

✅ **Alternative: Specify .env path explicitly:**
```javascript
require('dotenv').config({ path: '../../.env' });
```

---

## Supabase API Errors

### ❌ "Email address is invalid" Error

**Problem:**
```
User creation failed: Email address "user@example.com" is invalid
```

**Root Cause:**
Supabase may restrict certain email domains or formats.

**Solution:**
✅ **Use real email domains for testing:**

```javascript
// ❌ AVOID - Some test domains may be blocked
const email = 'test@example.com';
const email = 'user@test.local';

// ✅ PREFERRED - Use real domains
const email = 'test@gmail.com';
const email = 'user@yahoo.com';
```

### ❌ "Password should be at least 6 characters" Error

**Problem:**
Password validation fails during user creation.

**Root Cause:**
Supabase enforces minimum password length requirements.

**Solution:**
✅ **Ensure password meets requirements:**

```javascript
// ❌ TOO SHORT
const password = '123';

// ✅ VALID
const password = 'password123';  // 6+ characters
```

### ❌ Rate Limiting: "For security purposes, you can only request this after 59 seconds"

**Problem:**
Rapid successive calls with same email trigger rate limiting.

**Root Cause:**
Supabase implements security rate limiting for user creation.

**Solution:**
✅ **Use unique emails for testing:**

```javascript
// ❌ PROBLEMATIC - Reusing same email
const email = 'test@gmail.com';

// ✅ BETTER - Use timestamps for uniqueness
const timestamp = Date.now();
const email = `test-${timestamp}@gmail.com`;
```

✅ **Handle rate limiting gracefully:**
```javascript
try {
  const result = await createUserInSupabase(email, password);
} catch (error) {
  if (error.message.includes('For security purposes')) {
    // Treat as duplicate user scenario
    console.log('Rate limited - treating as existing user');
    // Handle appropriately for your use case
  }
}
```

---

## Testing Issues

### ❌ Jest Tests Failing with Module Import Errors

**Problem:**
```
Missing Supabase configuration. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.
```

**Root Cause:**
Environment variables not set before module import in Jest tests.

**Solution:**
✅ **Set environment variables before imports:**

```javascript
// ❌ WRONG - Import before setting env vars
const { createUserInSupabase } = require('./supabaseConfig');
process.env.DB_HOST = 'test-host';

// ✅ CORRECT - Set env vars before import
process.env.DB_HOST = process.env.DB_HOST || 'test-host';
process.env.DB_ANON_KEY = process.env.DB_ANON_KEY || 'test-key';
const { createUserInSupabase } = require('./supabaseConfig');
```

### ❌ Test Timeouts with Real API Calls

**Problem:**
Jest tests timeout when making real Supabase API calls.

**Solution:**
✅ **Increase test timeouts:**

```javascript
test('should create user', async () => {
  // Test implementation
}, 10000); // 10 second timeout for network calls
```

### ❌ Email Case Sensitivity in Tests

**Problem:**
```
Expected: "User@Gmail.com"
Received: "user@gmail.com"
```

**Root Cause:**
Supabase normalizes email addresses to lowercase.

**Solution:**
✅ **Expect lowercase emails in tests:**

```javascript
// ❌ WRONG - Exact case matching
expect(result.email).toBe(email);

// ✅ CORRECT - Account for normalization
expect(result.email).toBe(email.toLowerCase());
```

### ❌ Timestamp Precision Mismatches

**Problem:**
```
Expected: "2025-07-20T19:59:30.528361Z"
Received: "2025-07-20T19:59:30.528Z"
```

**Root Cause:**
Supabase returns timestamps with microsecond precision, JavaScript Date only supports milliseconds.

**Solution:**
✅ **Use regex patterns for timestamp validation:**

```javascript
// ❌ WRONG - Exact ISO string match
expect(new Date(result.createdAt).toISOString()).toBe(result.createdAt);

// ✅ CORRECT - Pattern matching
expect(result.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$/);
```

---

## Database Connection Issues

### ❌ "Connection refused" to Database

**Problem:**
Cannot connect to PostgreSQL database for application data.

**Diagnosis:**
```bash
# Test database connection
node -e "
const { Pool } = require('pg');
const pool = new Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD
});
pool.query('SELECT NOW()', (err, res) => {
  if (err) console.error('Connection failed:', err);
  else console.log('Connected! Time:', res.rows[0].now);
  pool.end();
});
"
```

**Solution:**
✅ **Verify all database environment variables:**
- `DB_HOST` - Should be database hostname (with `db.` prefix)
- `DB_PORT` - Should be `5432`
- `DB_NAME` - Should be `postgres`
- `DB_USER` - Database username
- `DB_PASSWORD` - Database password

---

## Common Development Issues

### ❌ Function Works in Manual Test but Fails in Jest

**Problem:**
Manual script works but Jest tests fail with same function.

**Root Cause:**
Different execution contexts have different environment variable loading.

**Solution:**
✅ **Ensure consistent environment setup:**

1. **Use same .env loading approach**
2. **Run both from same directory**
3. **Verify environment variables in both contexts**

### ❌ Intermittent Connection Failures

**Problem:**
Function works sometimes but fails randomly.

**Root Cause:**
Network instability or Supabase service interruptions.

**Solution:**
✅ **Implement retry logic for production:**

```javascript
async function createUserWithRetry(email, password, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await createUserInSupabase(email, password);
    } catch (error) {
      if (attempt === maxRetries || error.code === 'EMAIL_ALREADY_EXISTS') {
        throw error; // Don't retry on final attempt or user errors
      }
      
      console.log(`Attempt ${attempt} failed, retrying...`);
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
}
```

---

## Diagnostic Commands

### Environment Verification
```bash
# Test environment variable loading
node testEnvVariables.js

# Check .env file
ls -la .env && echo "File size: $(wc -c < .env) bytes"
```

### Network Connectivity
```bash
# Test API endpoint
curl -I https://zpwvrbabgegtyfnwdfjb.supabase.co

# Test DNS resolution
nslookup zpwvrbabgegtyfnwdfjb.supabase.co
```

### Function Testing
```bash
# Manual test script
node services/user-authentication-service/testSupabase.js

# Jest integration tests
npm test -- tests/services/user-authentication-service/supabase.test.js
```

---

## Getting Help

### Debug Information to Collect

When reporting issues, include:

1. **Environment Variables Status:**
   ```bash
   node testEnvVariables.js
   ```

2. **Network Connectivity:**
   ```bash
   curl -I https://your-project-id.supabase.co
   ```

3. **Error Details:**
   - Full error message and stack trace
   - Error code (if available)
   - Input parameters that caused the error

4. **Context:**
   - Operating system and Node.js version
   - Whether issue occurs in manual test, Jest, or both
   - Whether environment variables are loaded correctly

### Quick Health Check

Run this command to verify system health:
```bash
node -e "
require('dotenv').config();
console.log('✅ Environment loaded');
console.log('DB_HOST exists:', !!process.env.DB_HOST);
console.log('DB_ANON_KEY exists:', !!process.env.DB_ANON_KEY);
const host = process.env.DB_HOST?.replace(/^db\./, '');
console.log('API URL would be: https://' + host);
"
```

---

---

## Tool Registry Service Issues

### ❌ "Invalid or expired token" Error

**Problem:**
```
{
  "error": "Invalid or expired token"
}
```

**Root Cause:**
JWT authentication failing in Tool Registry Service due to incorrect Supabase client setup or expired tokens.

**Diagnosis:**
```bash
# Check if service is running
curl http://localhost:3002/health

# Test with fresh token
curl -H "Authorization: Bearer $(curl -s -X POST http://localhost:3001/api/v1/auth/login -H "Content-Type: application/json" -d '{"email":"merchant@test.com","password":"Test123!"}' | grep -o '"token":"[^"]*"' | cut -d'"' -f4)" http://localhost:3002/api/v1/tools
```

**Solution:**
✅ **Ensure proper JWT session setup in Tool Registry Service:**

```javascript
// ✅ CORRECT - Set JWT session for authenticated queries
const authSupabase = createClient(supabaseUrl, supabaseAnonKey);
const { data: sessionData, error: sessionError } = await authSupabase.auth.setSession({
  access_token: token,
  refresh_token: token
});

if (sessionError) {
  throw new Error('Failed to authenticate session');
}

// Now use authSupabase for database queries
const { data: tools, error } = await authSupabase.from('tools').select('*');
```

### ❌ "Merchant record not found" Error

**Problem:**
```
[AUTH] Merchant record not found for user_id: 7f6ba12a-1e8f-47f5-a0f7-2c2ec6db07a5
```

**Root Cause:**
Incorrect schema relationship chain or missing merchant records in database.

**Diagnosis:**
```bash
# Check if merchant records exist
node -e "
const { createClient } = require('@supabase/supabase-js');
// ... get JWT token and set session ...
const result = await authSupabase.from('merchants').select('*');
console.log('Merchants:', result.data.length);
"
```

**Solution:**
✅ **Verify correct schema relationship chain:**

```
auth.users.id → public.users.auth_id → public.users.uid → public.merchants.user_id → public.merchants.id → public.tools.merchant_id
```

✅ **Always use Schemas.md as source of truth for relationships**

### ❌ Row Level Security (RLS) Policy Violations

**Problem:**
```
{
  "code": "42501",
  "message": "new row violates row-level security policy for table \"tools\""
}
```

**Root Cause:**
RLS policies blocking tool creation because ownership chain cannot be verified.

**Solution:**
✅ **Ensure RLS policies allow the correct ownership chain validation**
✅ **Use JWT-authenticated Supabase client for all database operations**
✅ **Verify merchant_id matches the authenticated user's merchant record**

---

## Frontend Integration Issues

### ❌ "Network error" in Dashboard

**Problem:**
React frontend shows "Network error" when trying to fetch tools.

**Root Cause:**
Tool Registry Service not running or wrong API URL configuration.

**Diagnosis:**
```bash
# Check if Tool Registry Service is running
curl http://localhost:3002/health

# Check frontend environment variables
grep REACT_APP_TOOL_REGISTRY_URL web-portal/.env
```

**Solution:**
✅ **Ensure Tool Registry Service is running on port 3002**
✅ **Set correct environment variable:**

```bash
# In web-portal/.env or in start script
REACT_APP_TOOL_REGISTRY_URL=http://localhost:3002
```

### ❌ TypeScript Compilation Errors

**Problem:**
```
TS2322: Type 'AuthHeader' is not assignable to type 'AxiosHeaders'
```

**Root Cause:**
Incorrect type usage in axios headers configuration.

**Solution:**
✅ **Destructure header objects correctly:**

```typescript
// ❌ WRONG
const response = await axios.get(url, { headers: authHeader });

// ✅ CORRECT
const response = await axios.get(url, { headers: { ...authHeader } });
```

---

## Service Management Issues

### ❌ Port Conflicts Between Services

**Problem:**
Services fail to start due to port conflicts or old processes.

**Diagnosis:**
```bash
# Check what's running on ports
lsof -i :3000 -i :3001 -i :3002
```

**Solution:**
✅ **Use service management scripts:**

```bash
# Kill all processes and restart services
./start-all-services.sh

# Stop all services
./stop-all-services.sh
```

✅ **Always restart services after code changes:**
- Prevents issues with old processes running outdated code
- Ensures latest environment variables are loaded
- Clears any session state issues

### ❌ "Route not found" Errors

**Problem:**
API calls return 404 "Route not found" errors.

**Root Cause:**
Incorrect API endpoint URLs or services running on wrong ports.

**Solution:**
✅ **Verify service endpoints:**

```bash
# User Auth Service (port 3001)
curl http://localhost:3001/health

# Tool Registry Service (port 3002)  
curl http://localhost:3002/health

# React Frontend (port 3000)
curl http://localhost:3000
```

✅ **Check environment variables in services:**
- `USER_AUTH_PORT=3001`
- `TOOL_REGISTRY_PORT=3002`
- `REACT_APP_API_BASE_URL=http://localhost:3001`
- `REACT_APP_TOOL_REGISTRY_URL=http://localhost:3002`

---

## Database Query Issues

### ❌ Empty Results When Data Exists

**Problem:**
Database queries return empty arrays even when data should exist.

**Root Cause:**
Incorrect column names or relationships in queries.

**Diagnosis:**
```bash
# Test query with direct database access
node debug-query.js
```

**Solution:**
✅ **Always verify column names in Schemas.md before writing queries**
✅ **Test queries with JWT authentication properly set up**
✅ **Use correct foreign key relationships:**

- `public.users.uid` (NOT `id`) for user references
- `public.merchants.id` for merchant references in tools table
- `public.merchants.user_id` references `public.users.uid`

### ❌ Schema Relationship Confusion

**Problem:**
Confusion between different ID columns in related tables.

**Solution:**
✅ **Reference chain for tools queries:**

1. JWT token provides `auth.users.id`
2. Look up `public.users` where `auth_id = auth.users.id`
3. Get `public.users.uid`
4. Look up `public.merchants` where `user_id = public.users.uid`  
5. Get `public.merchants.id`
6. Query `public.tools` where `merchant_id = public.merchants.id`

✅ **Always use Schemas.md as the single source of truth**

---

## Development Best Practices

### ✅ Service Restart Protocol

**When debugging issues:**
1. Always kill all running Node.js processes
2. Restart all services using `./start-all-services.sh`
3. This prevents issues with:
   - Old processes running outdated code
   - Stale environment variables
   - Session state conflicts
   - Port conflicts

### ✅ Schema Verification

**Before writing any database query:**
1. Check `Build Documentation/Schemas.md`
2. Verify table names and column names
3. Confirm foreign key relationships
4. Never rely on old tickets or assumptions

### ✅ JWT Authentication Testing

**For testing API endpoints:**
```bash
# Get fresh token
TOKEN=$(curl -s -X POST http://localhost:3001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"merchant@test.com","password":"Test123!"}' \
  | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

# Test API endpoint
curl -H "Authorization: Bearer $TOKEN" http://localhost:3002/api/v1/tools
```

---

## Related Documentation

- [API Reference](./API-Reference.md) - Complete function specifications and usage examples
- [Project Structure](./structure.md) - File organization and component details
- [Sprint Progress](./Sprint-Progress.md) - Development timeline and implementation notes
- [Database Schema](./Schemas.md) - Database structure and configuration

**Next Update:** This document will be updated as new issues are encountered and resolved in future tickets. 