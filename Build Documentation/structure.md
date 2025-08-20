# Project Structure Documentation

**Version:** 1.3  
**Last Updated:** August 13, 2025  
**Sprint:** 2 - Tool Registry Service & Merchant Dashboard

## Overview

This document maintains a comprehensive record of the Agent Tool Marketplace project structure, documenting all folders and files as they are added during development. The project follows a monorepo architecture with npm workspaces.

**Related Documentation:**
- [Sprint Progress](./Sprint-Progress.md) - Development progress tracking
- [API Reference](./API-Reference.md) - Function specifications and usage
- [Troubleshooting](./Troubleshooting.md) - Issue resolution guide

---

## Root Directory Structure

```
/Paygentic (project-root)
├── /services/                    # Backend services (current & future)
├── /packages/                    # Shared libraries and utilities
├── /web-portal/                  # React frontend application
├── /tests/                       # Test files and configuration
├── /Documentation/               # Project documentation
├── /Build Documentation/         # Build and development documentation
│   ├── structure.md              # Project structure documentation
│   └── Schemas.md                # Database schema documentation
├── /Tickets/                     # Sprint planning and tickets
├── package.json                  # Root workspace configuration
├── package-lock.json             # Dependency lock file
├── README.md                     # Project setup and usage guide
├── .gitignore                    # Git exclusion patterns
└── node_modules/                 # NPM dependencies
```

---

## Root Level Files

### `package.json`
- **Purpose:** Root workspace configuration for the monorepo
- **Key Features:**
  - Defines workspaces for `services/*` and `packages/*`
  - Contains project-wide scripts (`start`, `dev`, `test`)
  - Manages shared development dependencies (Jest, Nodemon)
  - Sets Node.js version requirements (>=18.0.0)
- **Created in:** Ticket 0001
- **Dependencies:** Jest, Nodemon

### `README.md`
- **Purpose:** Main project documentation and setup guide
- **Contents:**
  - Project overview and structure explanation
  - Installation and running instructions
  - Available endpoints documentation
  - Configuration guidance
  - Development workflow information
- **Created in:** Ticket 0001

### `.gitignore`
- **Purpose:** Git exclusion patterns for Node.js projects
- **Excludes:** node_modules, .env files, logs, build outputs, IDE files, OS files
- **Created in:** Ticket 0001

---

## Services Directory

The `/services/` directory contains backend microservices that can be developed independently but share common utilities.

### `/services/user-authentication-service/`

**Purpose:** Core user authentication and management service  
**Created in:** Ticket 0001

#### Files:
- **`package.json`**
  - Service-specific dependencies and scripts
  - Dependencies: Express.js, dotenv, cors, helmet, morgan
  - Dev dependencies: nodemon, jest, supertest
  
- **`index.js`** (Main application file)
  - Express.js server setup with security middleware
  - Health check endpoint: `GET /health`
  - API status endpoint: `GET /api/v1/status`
  - Comprehensive error handling and logging
  - Port configuration (default: 3000)

#### API Endpoints:
- `GET /health` - Service health check
- `GET /api/v1/status` - Service status and version info

#### Key Features:
- **Security:** Helmet middleware for security headers
- **CORS:** Cross-origin resource sharing enabled
- **Logging:** Morgan middleware for HTTP request logging
- **Error Handling:** Comprehensive error middleware with dev/prod modes
- **Body Parsing:** JSON and URL-encoded parsing configured

---

### `/services/tool-registry-service/`

**Purpose:** Tool management and validation service for merchants  
**Created in:** Ticket 2002

#### Files:
- **`package.json`**
  - Service-specific dependencies and scripts
  - Dependencies: Express.js, cors, helmet, morgan, dotenv, axios, @supabase/supabase-js
  - Dev dependencies: nodemon, jest, supertest
  
- **`index.js`** (Main application file)
  - Express.js server setup with security middleware
  - Health check endpoint: `GET /health`
  - Tool creation endpoint: `POST /api/v1/tools`
  - JWT authentication integration with Supabase
  - Comprehensive validation and error handling
  - Port configuration (default: 3002)

#### API Endpoints:
- `GET /health` - Service health check
- `POST /api/v1/tools` - Create new tool (Merchant-only, RBAC protected)
- `GET /api/v1/tools` - Get list of tools for authenticated merchant (RBAC protected)
- `GET /api/v1/tools/{id}` - Get specific tool by ID (Owner verification required)
- `PUT /api/v1/tools/{id}` - Update existing tool (Owner verification required)

#### Key Features:
- **Security:** Helmet middleware, CORS, JWT authentication
- **RBAC:** Role-based access control (MERCHANT account type only)
- **Validation:** Comprehensive request validation with detailed error messages
- **Endpoint Verification:** HTTP HEAD requests to verify endpoint reachability
- **Database Integration:** PostgreSQL with foreign key constraints
- **Input Sanitization:** Automatic trimming of string fields
- **Error Handling:** Structured error responses with appropriate HTTP status codes

#### Authentication Flow:
1. Bearer token extraction from Authorization header
2. JWT verification using Supabase client
3. User profile retrieval with merchant validation
4. Automatic merchant_id assignment for tool creation

#### Validation Rules:
- **name:** Required string, max 255 characters, non-empty after trim
- **description:** Required string, max 10,000 characters, non-empty after trim
- **endpoint_url:** Required string, max 500 characters, valid URL format, reachable via HEAD request
- **price_per_call:** Required number, min 0.0001, max 999999.9999

---

## Packages Directory

The `/packages/` directory contains shared libraries and utilities used across multiple services.

### `/packages/common/`

**Purpose:** Shared utilities, types, and helper functions  
**Created in:** Ticket 0001

#### Files:
- **`package.json`**
  - Common package configuration
  - Dev dependencies: Jest for testing
  
- **`index.js`** (Main exports)
  - Account types enumeration (`MERCHANT`, `AGENT`, `ADMIN`)
  - Response helper functions (success, error, validation error)
  - Common validation utilities (email, account type, password)

#### Exported Utilities:
- **`ACCOUNT_TYPES`** - Enumeration of user account types
- **`responses`** - Standardized API response helpers
- **`validators`** - Common validation functions

### `/packages/database/`

**Purpose:** Database utilities, connection management, and migration system  
**Created in:** Ticket 0002

#### Files:
- **`package.json`**
  - Database package configuration
  - Dependencies: pg (PostgreSQL client), uuid, dotenv
  - Dev dependencies: Jest for testing
  
- **`index.js`** (Main exports)
  - Database connection pool and utilities
  - Query execution helpers
  - Connection testing functions
  
- **`config.js`** (Database configuration)
  - PostgreSQL connection pool setup
  - Environment variable configuration
  - Connection event handling
  
- **`README.md`** (Database documentation)
  - Setup instructions and environment variables
  - Schema documentation
  - Usage examples

#### Directory Structure:
```
/packages/database/
├── /migrations/
│   └── 001_create_core_schema.js    # Core schema migration
├── /scripts/
│   └── migrate.js                   # Migration runner system
├── config.js                       # Database configuration
├── index.js                        # Main exports
├── package.json                    # Package configuration
└── README.md                       # Documentation
```

#### Database Schema:
- **`users`** - Core user identity (id, email, name, dob, account_type, email_verified)
- **`merchants`** - Tool maker data (user_id, trust_rating)
- **`agents`** - Agent developer data (user_id, balance)
- **`tools`** - Tool registry and metadata (id, merchant_id, name, description, endpoint_url, price_per_call, listing_status, average_rating)
- **`migrations`** - Migration tracking table (auto-created)

#### Migration Commands:
- `npm run db:migrate` - Run all pending migrations
- `npm run db:migrate:up` - Run migrations up
- `npm run db:migrate:down` - Rollback last migration

#### Key Features:
- **Connection Pooling:** Efficient PostgreSQL connection management
- **Migration System:** Automated schema versioning and rollback
- **Transaction Support:** Safe migration execution with rollback
- **Environment Configuration:** Flexible database connection setup

---

## Web Portal Directory

The `/web-portal/` directory contains the React frontend application that provides the graphical interface for all human interaction with the marketplace.

### `/web-portal/`

**Purpose:** React-based frontend application for the marketplace  
**Created in:** Ticket 2006

#### Directory Structure:
```
/web-portal/
├── /src/
│   ├── /components/              # Reusable UI components
│   ├── /pages/                   # Top-level page components
│   ├── /services/                # API call logic
│   ├── /hooks/                   # Custom hooks
│   ├── /contexts/                # State management
│   ├── App.tsx                   # Main application component
│   ├── index.tsx                 # Application entry point
│   └── ... (other React files)
├── /public/                      # Static assets
├── package.json                  # Frontend dependencies and scripts
├── tsconfig.json                 # TypeScript configuration
└── README.md                     # Frontend documentation
```

#### Key Features:
- **Framework:** React 19.1.1 with TypeScript
- **Routing:** React Router DOM for navigation
- **HTTP Client:** Axios for API communication
- **Testing:** React Testing Library and Jest
- **Build System:** Create React App (CRA) tooling

#### Dependencies:
- **Core:** React, React DOM, TypeScript
- **Routing:** react-router-dom
- **HTTP:** axios
- **Testing:** @testing-library/react, @testing-library/jest-dom
- **Types:** @types/react, @types/react-dom, @types/react-router-dom

#### Available Scripts:
- `npm start` - Development server (default port: 3000)
- `npm run build` - Production build
- `npm test` - Run tests
- `npm run eject` - Eject from CRA (irreversible)

#### Implemented Services:

##### Authentication Service (Ticket 2007)
**File:** `/src/services/authService.ts`
**Purpose:** Frontend authentication service and JWT token management

**Functions:**
- `login(email, password)` - Authenticates user and stores JWT token
- `logout()` - Removes JWT token from localStorage
- `getToken()` - Retrieves JWT token from storage
- `isAuthenticated()` - Checks if user has valid token
- `getAuthHeader()` - Returns formatted authorization header for API calls

**Features:**
- ✅ TypeScript interfaces for type safety
- ✅ Comprehensive error handling with meaningful messages
- ✅ LocalStorage token management with key `paygentic_auth_token`
- ✅ Environment variable support for API base URL
- ✅ Integration with backend `POST /api/v1/auth/login` endpoint
- ✅ Structured logging for debugging

**Testing:**
- **Test File:** `/src/services/authService.test.ts`
- **Coverage:** 19 comprehensive tests covering all functions and error scenarios
- **Test Strategy:** Jest with mocked axios and localStorage
- ✅ Input validation tests
- ✅ Success scenario tests
- ✅ Error handling tests (401, 400, 500, network errors)
- ✅ Token storage/retrieval tests
- ✅ Integration scenario tests

#### Implemented Contexts:

##### Authentication Context (Ticket 2008)
**File:** `/src/contexts/AuthContext.js`
**Purpose:** Global authentication state management using React Context

**Context Values:**
- `isAuthenticated` - Boolean indicating authentication status
- `user` - User object with email and token information
- `login(email, password)` - Function that calls authService and updates state
- `logout()` - Function that calls authService and clears state
- `isLoading` - Boolean for loading states during operations
- `error` - Error message string when operations fail
- `clearError()` - Function to clear error state

**Features:**
- ✅ Session persistence across page reloads via token checking
- ✅ Integration with existing authService from Ticket 2007
- ✅ Comprehensive error handling and loading states
- ✅ Custom `useAuth` hook for easy context consumption
- ✅ Proper error boundary for usage outside provider

**Testing:**
- **Test File:** `/src/contexts/AuthContext.test.js`
- **Coverage:** 13 comprehensive tests covering all context functionality
- **Test Strategy:** React Testing Library with mocked authService
- ✅ Initialization scenarios (with/without tokens)
- ✅ Login/logout functionality and state updates
- ✅ Error handling and persistence
- ✅ Integration scenarios and edge cases

#### Implemented Components:

##### Protected Route Component (Ticket 2009)
**File:** `/src/components/ProtectedRoute.js`
**Purpose:** Reusable component that wraps routes to ensure only authenticated users can access them

**Functionality:**
- `children` - React components/routes to render if authenticated
- Uses `useAuth()` hook from AuthContext to check authentication status
- Uses `useLocation()` and `<Navigate>` from react-router for redirects
- Shows loading state while authentication is being verified
- Redirects unauthenticated users to `/login` page with location state preservation

**Features:**
- ✅ Integration with AuthContext for authentication state
- ✅ React Router v7 compatibility with correct imports from 'react-router'
- ✅ Loading state handling during authentication checks
- ✅ Location preservation for post-login redirects using Navigate state
- ✅ Support for multiple children and nested components
- ✅ Graceful handling of edge cases (null children, empty fragments)

**Usage Pattern:**
```jsx
<ProtectedRoute>
  <YourProtectedComponent />
</ProtectedRoute>
```

**Testing:**
- **Test File:** `/src/components/ProtectedRoute.test.js`
- **Coverage:** 10 comprehensive tests covering all component functionality
- **Test Strategy:** React Testing Library with mocked authService and MemoryRouter
- ✅ Authentication scenarios (authenticated vs unauthenticated users)
- ✅ Redirect behavior verification with proper Router context
- ✅ Content rendering tests with multiple and nested children
- ✅ Edge cases and integration with AuthContext
- ✅ Location preservation and props handling tests

##### Login Page (Ticket 2010)
**Files:**
- `web-portal/src/pages/LoginPage.js` — Email/password login form wired to `AuthContext`
- `web-portal/src/pages/LoginPage.css` — Styling for the login page
- `web-portal/src/pages/LoginPage.test.js` — Comprehensive tests (validation, errors, integration)
- `web-portal/src/pages/DashboardPage.js` — Temporary placeholder post-login landing page

**Routing and Auth Integration:**
- `web-portal/src/App.tsx` uses React Router v7 (`react-router`) with routes for `/login` and protected `/dashboard`
- `web-portal/src/components/ProtectedRoute.js` guards authenticated routes

**Notes:**
- React Router v7 imports are from `react-router` (not `react-router-dom`)
- Frontend calls backend `POST /api/v1/auth/login` via `src/services/authService.ts`

##### Dashboard Layout Component (Ticket 2011)
**File:** `/src/components/DashboardLayout.js`
**Purpose:** Main authenticated application layout shell

**Features:**
- ✅ Professional header with "Agent Tool Marketplace" branding
- ✅ Logout button integrated with AuthContext
- ✅ Collapsible sidebar for future navigation
- ✅ Responsive main content area
- ✅ Modern UI with accessibility features (ARIA labels)
- ✅ Toggle functionality for sidebar expand/collapse

**Testing:**
- **Test File:** `/src/components/DashboardLayout.test.js`
- **Coverage:** Comprehensive unit tests for layout functionality
- **Test Categories:** Rendering, logout functionality, sidebar toggle, integration tests

##### Merchant Dashboard Page (Ticket 2012)
**File:** `/src/pages/MerchantDashboardPage.tsx`
**Purpose:** Main dashboard page for merchants to view and manage their tools

**Functionality:**
- ✅ Fetches and displays merchant's tools using `toolService.getTools()`
- ✅ Professional UI with tool list display (name, price, status, dates)
- ✅ Loading state with spinner during API calls
- ✅ Error handling with user-friendly messages
- ✅ Empty state with guidance for new merchants
- ✅ "Create New Tool" button (ready for Ticket 2013)
- ✅ Integration with DashboardLayout and ProtectedRoute

**Features:**
- ✅ TypeScript implementation with full type safety
- ✅ Real-time tool data fetching with JWT authentication
- ✅ Responsive design with organized tool display
- ✅ Professional styling matching application theme
- ✅ Complete state management (loading, error, success, empty states)

**Testing:**
- **Test File:** `/src/pages/MerchantDashboardPage.test.tsx`
- **Coverage:** Comprehensive tests for all component states and functionality
- **Test Categories:** Loading states, error scenarios, tool display, empty states, integration

##### Tool Service (Ticket 2012)
**File:** `/src/services/toolService.ts`
**Purpose:** Frontend service for Tool Registry API integration

**Functions:**
- `getTools()` - Fetches all tools for authenticated merchant
- `createTool(toolData)` - Creates new tool (ready for future implementation)

**Features:**
- ✅ TypeScript interfaces for type safety (`Tool`, `CreateToolRequest`)
- ✅ JWT authentication with authorization headers
- ✅ Environment variable configuration (`REACT_APP_TOOL_REGISTRY_URL`)
- ✅ Comprehensive error handling with meaningful messages
- ✅ Integration with backend `GET /api/v1/tools` endpoint
- ✅ Structured logging for debugging

**Testing:**
- **Test File:** `/src/services/toolService.test.ts`
- **Coverage:** Comprehensive service testing with mocked axios
- **Test Categories:** Authentication, API calls, error handling, data transformation

##### Create Tool Modal Component (Ticket 2013)
**File:** `/src/components/CreateToolModal.tsx`
**Purpose:** Modal dialog for creating new tools with comprehensive form validation and API integration

**Component Interface:**
```typescript
interface CreateToolModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (tool: Tool) => void;
}
```

**Functionality:**
- **Modal Management:** Controlled by parent component (MerchantDashboardPage) with open/close state
- **Form Fields:** Tool name, description, endpoint URL, and price per call
- **Validation:** Real-time client-side validation with server-side error handling
- **API Integration:** Calls `toolService.createTool()` with proper authentication
- **Success Handling:** Refreshes parent tool list and closes modal automatically

**Form Features:**
- ✅ **Tool Name:** Required, max 255 characters, auto-trimmed input
- ✅ **Description:** Required, max 10,000 characters, resizable textarea
- ✅ **Endpoint URL:** Required, URL format validation, backend reachability check
- ✅ **Price per Call:** Required, $0.0001-$999,999.9999 range, currency formatting

**Validation System:**
- ✅ **Real-time Clearing:** Field errors clear when user starts typing
- ✅ **Multiple Display:** Errors shown both in summary list and individual field areas
- ✅ **Server Integration:** Backend validation errors displayed with proper formatting
- ✅ **Form State Management:** Prevents submission during loading states

**User Experience:**
- ✅ **Professional Design:** Consistent with application styling and responsive
- ✅ **Loading States:** Visual feedback during submission with disabled controls
- ✅ **Accessibility:** ARIA labels, keyboard navigation, screen reader support
- ✅ **Error Recovery:** Form re-enables after errors with clear messaging

**Modal Behavior:**
- ✅ **Backdrop Close:** Clicking outside modal closes it (when not submitting)
- ✅ **Escape Handling:** Close button and cancel action available
- ✅ **State Preservation:** Prevents accidental close during active submission
- ✅ **Form Reset:** Clean state on close and successful submission

**Testing:**
- **Test File:** `/src/components/CreateToolModal.test.tsx`
- **Coverage:** 23/23 comprehensive tests passing
- **Test Categories:** Modal rendering, form validation, API integration, error handling, accessibility
- **Integration Tests:** Complete tool creation flow with dashboard refresh
- **Error Scenarios:** Network errors, validation failures, server errors

**TypeScript Implementation:**
- ✅ **Full Type Safety:** Proper interfaces for all props, state, and event handlers
- ✅ **Error Handling:** Proper handling of unknown error types in catch blocks
- ✅ **Event Types:** Correctly typed React event handlers for forms and clicks
- ✅ **Integration:** Seamless integration with existing TypeScript toolService

#### Future Components (Planned):
- **Tool Edit Modal:** Interface for updating existing tools
- **Agent Developer Dashboard:** Agent account management
- **HITL Confirmation Modal:** Real-time approval interface
- **Administrator Dashboard:** Platform oversight interface

---

## Tests Directory

The `/tests/` directory contains all test files organized to mirror the structure of the source code.

### Directory Structure

```
/tests/
├── /services/                 # Tests for backend services
│   └── /user-authentication-service/
│       └── health.test.js     # Health endpoint tests
├── /packages/                 # Tests for shared packages
│   ├── /common/
│   │   ├── validators.test.js # Validation function tests
│   │   └── responses.test.js  # Response helper tests
│   └── /database/
│       └── migration.test.js  # Database migration tests
├── jest.config.js            # Jest configuration
├── setup.js                  # Global test setup
└── README.md                 # Test documentation
```

**Created in:** Ticket 0001

### Test Files

- **`tests/services/user-authentication-service/health.test.js`**
  - Integration tests for health and status endpoints
  - Uses supertest for HTTP endpoint testing
  - Tests success cases and error handling

- **`tests/packages/common/validators.test.js`**
  - Unit tests for validation functions
  - Tests email, account type, and password validation
  - Covers both valid and invalid input cases

- **`tests/packages/common/responses.test.js`**
  - Unit tests for response helper functions
  - Tests success, error, and validation error responses
  - Verifies proper response structure

- **`tests/packages/database/migration.test.js`**
  - Tests for database migration system
  - Validates schema creation and migration tracking
  - Tests foreign key constraints and table structure

### Configuration Files

- **`jest.config.js`** - Jest configuration with coverage settings
- **`setup.js`** - Global test setup and environment configuration
- **`README.md`** - Comprehensive testing documentation

### Test Categories

- **Unit Tests:** Test individual functions and components
- **Integration Tests:** Test API endpoints and service interactions
- **Coverage:** Target 80%+ code coverage for all services and packages

---

## Development Infrastructure

### Package Management
- **Architecture:** NPM workspaces monorepo
- **Node.js Version:** >=18.0.0
- **Package Manager:** NPM >=9.0.0

### Scripts Available
- `npm start` - Production mode (runs user-authentication-service)
- `npm run dev` - Development mode with nodemon
- `npm test` - Run all tests using Jest
- `npm run test:watch` - Run tests in watch mode
- `npm run test:coverage` - Run tests with coverage reports
- `npm run install-all` - Install dependencies for all workspaces
- `npm run db:migrate` - Run database migrations
- `npm run db:migrate:up` - Run migrations up
- `npm run db:migrate:down` - Rollback last migration

### Testing Framework
- **Framework:** Jest
- **Coverage:** Configured for all workspaces
- **Integration Testing:** Supertest for API endpoint testing

---

## Future Expansion Areas

### Planned Services (Future Sprints)
- `/services/tool-discovery-service/` - Tool search and discovery API
- `/services/payment-billing-service/` - Internal ledger and transactions
- `/services/hitl-orchestration-service/` - Human-in-the-loop workflow
- `/services/rating-trust-service/` - Rating and trust calculations

### Planned Packages (Future Sprints)
- `/packages/database/` - Database models and migrations
- `/packages/auth/` - Authentication utilities and middleware
- `/packages/mcp/` - Model Context Protocol implementations
- `/packages/logging/` - Structured logging utilities

---

## Documentation Status

### Completed Documentation
- [x] Project structure setup (this document)
- [x] README.md with setup instructions
- [x] API endpoint documentation (basic)
- [x] Test directory structure and documentation
- [x] Database schema documentation (Schemas.md)

### Future Documentation Tasks
- [x] Database schema documentation (see [Schemas.md](./Schemas.md))
- [x] API specification for authentication functions (see [API-Reference.md](./API-Reference.md))
- [x] Troubleshooting guide (see [Troubleshooting.md](./Troubleshooting.md))
- [ ] OpenAPI/Swagger specification for REST endpoints
- [ ] Deployment guide
- [ ] Environment configuration guide

---

## Version History

| Version | Date | Sprint/Ticket | Changes |
|---------|------|---------------|---------|
| 1.0 | July 9, 2025 | Sprint 1 / Ticket 0001 | Initial project structure setup, user-authentication-service, common utilities |
| 1.1 | July 9, 2025 | Sprint 1 / Ticket 0001 | Added tests directory with Jest configuration, comprehensive test suite (15 tests, 84% coverage), updated package.json scripts |
| 1.2 | July 9, 2025 | Sprint 1 / Ticket 0002 | Created database package with PostgreSQL connection, migration system, and core schema (users, merchants, agents tables). Added 6 database tests (21 total tests passing) |
| 1.3 | July 9, 2025 | Sprint 1 / Ticket 0002 | Added comprehensive database schema documentation (Schemas.md) with ERD, constraints, relationships, and future evolution plans |
| 1.4 | July 15, 2025 | Sprint 1 / Ticket 0003 | Added initial admin and installed supabase dependencies |
| 1.5 | July 20, 2025 | Sprint 1 / Ticket 0004 | Added Supabase user creation function, comprehensive API documentation, troubleshooting guide |
| 1.6 | July 20, 2025 | Documentation Reorganization | Created Sprint-Progress.md, added cross-references between all documentation files, updated README.md environment variables |
| 1.7 | July 21, 2025 | Sprint 1 / Ticket 0006 | Added user registration REST endpoint (POST /api/v1/auth/register), comprehensive test suite (16 tests), two-schema database integration, rate limiting handling, updated all documentation |
| 1.8 | July 22, 2025 | Sprint 1 / Ticket 0006 (Refactor) | Refactored registration to use Supabase database triggers for automatic record creation, simplified codebase (~50 lines removed), updated trust_rating default to -1.0, improved test suite termination and cleanup |
| 1.9 | August 1, 2025 | Sprint 2 / Ticket 2002 | Added Tool Registry Service with POST /api/v1/tools endpoint, comprehensive validation, RBAC security, endpoint reachability checks, and full test suite (28 tests). Created tools table migration and database integration. |
| 1.10 | August 1, 2025 | Sprint 2 / Tickets 2003 & 2004 | Added GET /api/v1/tools and GET /api/v1/tools/{id} endpoints for tool retrieval (18 tests), and PUT /api/v1/tools/{id} endpoint for tool updates (27 tests). Complete CRUD operations with ownership verification and comprehensive test coverage (73 total tool registry tests). |
| 1.11 | August 1, 2025 | Sprint 2 / Ticket 2006 | Initialize React Frontend Application - Created web-portal directory with React 19.1.1 + TypeScript, implemented required folder structure (components, pages, services, hooks, contexts), installed axios and react-router-dom dependencies. Frontend foundation ready for merchant dashboard development. |
| 1.12 | August 1, 2025 | Sprint 2 / Ticket 2007 | Added Frontend Authentication Service - Created authService.ts with comprehensive JWT token management, localStorage integration, backend API communication, and complete test suite (19 tests). TypeScript implementation with error handling. |
| 1.13 | August 1, 2025 | Sprint 2 / Ticket 2008 | Implemented Global Authentication Context - Created AuthContext.js with React Context for global state management, session persistence, custom useAuth hook, and comprehensive test suite (13 tests). Complete integration with authService. |
| 1.14 | August 1, 2025 | Sprint 2 / Ticket 2009 | Created Protected Route Component - Implemented ProtectedRoute.js with authentication-based route protection, React Router v7 compatibility, location preservation for post-login redirects, and comprehensive test suite (10 tests). Complete integration with AuthContext and router navigation. |
| 1.15 | August 13, 2025 | Sprint 2 / Ticket 2011 | Built Main Dashboard Layout Component - Created DashboardLayout.js with professional header, collapsible sidebar, logout functionality, and comprehensive test suite. Complete integration with AuthContext for authentication flow. |
| 1.16 | August 13, 2025 | Sprint 2 / Ticket 2012 | Implemented Merchant Dashboard Page with Tool List - Created MerchantDashboardPage.tsx and toolService.ts with complete tool listing functionality, JWT authentication, TypeScript implementation, comprehensive state management, and full test coverage (107/107 tests passing). Major technical issues resolved including JWT authentication flow, database schema relationships, and RLS integration. |
| 1.17 | August 13, 2025 | Sprint 2 / Ticket 2013 | Implemented Create Tool Modal and Form - Created CreateToolModal.tsx with comprehensive form validation, TypeScript implementation, API integration with toolService.createTool(), modal state management, accessibility features, and complete test suite (23/23 tests passing). Updated MerchantDashboardPage with modal integration and tool list refresh functionality. |
---

**Last Updated:** Updated in Ticket 2013 with Create Tool Modal implementation and comprehensive form validation.

---

## Related Documentation

For detailed sprint progress tracking, see [Sprint Progress](./Sprint-Progress.md).  
For API function details, see [API Reference](./API-Reference.md).  
For issue resolution, see [Troubleshooting Guide](./Troubleshooting.md).