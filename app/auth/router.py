from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.email_service import send_otp_email
from app.auth.jwt import create_access_token
from app.auth.otp import generate_otp, get_otp_expiration
from app.auth.schemas import (
    RegisterRequest,
    RegisterResponse,
    VerifyOTPRequest,
    VerifyOTPResponse,
    LoginRequest,
    LoginResponse,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    VerifyResetOTPRequest,
    VerifyResetOTPResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
)
from app.auth.security import hash_password, verify_password
from app.database.connection import get_db
from app.database.repositories import UserRepository


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    # --------------------------------------------------------
    # Check email
    # --------------------------------------------------------

    if user_repository.get_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered.",
        )

    # --------------------------------------------------------
    # Check username
    # --------------------------------------------------------

    if user_repository.get_by_username(request.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already registered.",
        )

    # --------------------------------------------------------
    # Hash password
    # --------------------------------------------------------

    password_hash = hash_password(request.password)

    # --------------------------------------------------------
    # Create user
    # --------------------------------------------------------

    user = user_repository.create(
        username=request.username,
        email=request.email,
        password_hash=password_hash,
    )

    # --------------------------------------------------------
    # Generate OTP
    # --------------------------------------------------------

    otp = generate_otp()

    # Store ONLY hashed OTP
    user.otp_hash = hash_password(otp)

    # Store expiration time
    user.otp_expires_at = get_otp_expiration()

    # --------------------------------------------------------
    # Send OTP BEFORE commit
    # --------------------------------------------------------

    try:
        send_otp_email(
            to_email=user.email,
            otp=otp,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email.",
        ) from exc

    # --------------------------------------------------------
    # Save user ONLY after email is sent successfully
    # --------------------------------------------------------

    try:
        db.commit()
        db.refresh(user)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user.",
        ) from exc

    return RegisterResponse(
        message="Registration successful. Please verify your email.",
        email=user.email,
    )


# ============================================================
# RESEND VERIFICATION OTP
# ============================================================

@router.post(
    "/resend-verification-otp",
)
def resend_verification_otp(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = user_repository.get_by_email(request.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # --------------------------------------------------------
    # Check active account
    # --------------------------------------------------------

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # --------------------------------------------------------
    # Check if already verified
    # --------------------------------------------------------

    if user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already verified.",
        )

    # --------------------------------------------------------
    # Generate new verification OTP
    # --------------------------------------------------------

    otp = generate_otp()

    user.otp_hash = hash_password(otp)
    user.otp_expires_at = get_otp_expiration()

    # --------------------------------------------------------
    # Send OTP BEFORE commit
    # --------------------------------------------------------

    try:
        send_otp_email(
            to_email=user.email,
            otp=otp,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email.",
        ) from exc

    # --------------------------------------------------------
    # Save OTP
    # --------------------------------------------------------

    try:
        db.commit()
        db.refresh(user)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save verification OTP.",
        ) from exc

    return {
        "message": "Verification OTP has been sent successfully.",
        "email": user.email,
    }


# ============================================================
# FORGOT PASSWORD
# ============================================================

@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
)
def forgot_password(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    user = user_repository.get_by_email(request.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # --------------------------------------------------------
    # Generate OTP
    # --------------------------------------------------------

    otp = generate_otp()

    user.otp_hash = hash_password(otp)
    user.otp_expires_at = get_otp_expiration()

    # --------------------------------------------------------
    # Send OTP BEFORE commit
    # --------------------------------------------------------

    try:
        send_otp_email(
            to_email=user.email,
            otp=otp,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send password reset email.",
        ) from exc

    # --------------------------------------------------------
    # Save OTP
    # --------------------------------------------------------

    try:
        db.commit()
        db.refresh(user)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save password reset OTP.",
        ) from exc

    return ForgotPasswordResponse(
        message="Password reset OTP has been generated."
    )


# ============================================================
# VERIFY REGISTRATION OTP
# ============================================================

@router.post(
    "/verify-otp",
    response_model=VerifyOTPResponse,
)
def verify_otp(
    request: VerifyOTPRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = user_repository.get_by_email(request.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # --------------------------------------------------------
    # Check if already verified
    # --------------------------------------------------------

    if user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already verified.",
        )

    # --------------------------------------------------------
    # Check OTP exists
    # --------------------------------------------------------

    if not user.otp_hash or not user.otp_expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active OTP found.",
        )

    # --------------------------------------------------------
    # Check OTP expiration
    # --------------------------------------------------------

    now = datetime.now(timezone.utc)

    if now > user.otp_expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired.",
        )

    # --------------------------------------------------------
    # Verify OTP
    # --------------------------------------------------------

    if not verify_password(
        request.otp,
        user.otp_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP.",
        )

    # --------------------------------------------------------
    # Verify user
    # --------------------------------------------------------

    user.is_verified = True

    # --------------------------------------------------------
    # Remove OTP after successful verification
    # --------------------------------------------------------

    user.otp_hash = None
    user.otp_expires_at = None

    # --------------------------------------------------------
    # Save changes
    # --------------------------------------------------------

    try:
        db.commit()
        db.refresh(user)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email.",
        ) from exc

    return VerifyOTPResponse(
        message="Email verified successfully.",
        email=user.email,
    )


# ============================================================
# VERIFY RESET PASSWORD OTP
# ============================================================

@router.post(
    "/verify-reset-otp",
    response_model=VerifyResetOTPResponse,
)
def verify_reset_otp(
    request: VerifyResetOTPRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = user_repository.get_by_email(request.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # --------------------------------------------------------
    # Check active account
    # --------------------------------------------------------

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # --------------------------------------------------------
    # Check OTP exists
    # --------------------------------------------------------

    if not user.otp_hash or not user.otp_expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active OTP found.",
        )

    # --------------------------------------------------------
    # Check OTP expiration
    # --------------------------------------------------------

    now = datetime.now(timezone.utc)

    if now > user.otp_expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired.",
        )

    # --------------------------------------------------------
    # Verify OTP
    # --------------------------------------------------------

    if not verify_password(
        request.otp,
        user.otp_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP.",
        )

    # --------------------------------------------------------
    # TEMPORARY RESET TOKEN
    # --------------------------------------------------------

    reset_token = f"reset-{user.id}"

    return VerifyResetOTPResponse(
        message="Reset OTP verified successfully.",
        reset_token=reset_token,
    )


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=LoginResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    user = user_repository.get_by_email(request.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email is not verified.",
        )

    if not verify_password(
        request.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    access_token = create_access_token(
        user_id=user.id,
        role=user.role,
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
    )


# ============================================================
# RESET PASSWORD
# ============================================================

@router.post(
    "/reset-password",
    response_model=ResetPasswordResponse,
)
def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    user_repository = UserRepository(db)

    # --------------------------------------------------------
    # Temporary reset token validation
    # --------------------------------------------------------

    if not request.reset_token.startswith("reset-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token.",
        )

    try:
        user_id = int(
            request.reset_token.replace("reset-", "", 1)
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token.",
        )

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = user_repository.get_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # --------------------------------------------------------
    # Update password
    # --------------------------------------------------------

    user.password_hash = hash_password(
        request.new_password
    )

    # --------------------------------------------------------
    # Clear reset OTP data
    # --------------------------------------------------------

    user.otp_hash = None
    user.otp_expires_at = None

    # --------------------------------------------------------
    # Save changes
    # --------------------------------------------------------

    try:
        db.commit()
        db.refresh(user)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password.",
        ) from exc

    return ResetPasswordResponse(
        message="Password has been reset successfully.",
    )