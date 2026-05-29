type ValidationBannerProps = {
  message: string | null;
};

export function ValidationBanner({ message }: ValidationBannerProps) {
  if (!message) {
    return null;
  }

  return (
    <div className="validation-banner" role="alert">
      {message}
    </div>
  );
}

