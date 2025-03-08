const port = process.env.PORT || 3000; // Ensure it's listening on the correct port.
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
