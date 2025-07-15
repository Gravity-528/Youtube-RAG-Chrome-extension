let popover: HTMLElement | null = null;
let currentSelectedText: string = "";
let currentExplanation: string = "";
let isLoading: boolean = false;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "videoId") {
    sendResponse({ status: "success", videoId: request.videoId });

  } else if (request.action === "others") {
    document.removeEventListener("mouseup", handleMouseUp);
    document.addEventListener("mouseup", handleMouseUp);
    sendResponse({ status: "success", url: request.url });
    return true;
  } else {
    sendResponse({ status: "error", message: "Unknown action" });
  }
});

function handleMouseUp(event: MouseEvent) {
  const target = event.target as Node;

  if (popover && popover.contains(target)) return;

  const selection = window.getSelection();
  const selectedText = selection?.toString().trim() || "";

  if (selectedText.length > 0 && selectedText.length <= 400) {
    const range = selection!.getRangeAt(0).getBoundingClientRect();
    const x = range.left + window.scrollX;
    const y = range.bottom + window.scrollY + 10;

    currentSelectedText = selectedText;
    currentExplanation = "";
    isLoading = false;

    createOrUpdatePopover(currentSelectedText, x, y, currentExplanation, isLoading);
  } else {
    hidePopover();
  }
}

function hidePopover() {
  if (popover) {
    popover.remove();
    popover = null;
    currentSelectedText = "";
    currentExplanation = "";
    isLoading = false;
  }
}

function createOrUpdatePopover(
  text: string,
  x: number,
  y: number,
  explanation: string,
  isLoading: boolean
) {
  if (popover) popover.remove();

  popover = document.createElement("div");
  popover.id = "popover";
  Object.assign(popover.style, {
    position: "absolute",
    left: `${x}px`,
    top: `${y}px`,
    backgroundColor: "black",
    border: "2px solid darkorange",
    borderRadius: "8px",
    padding: "10px",
    zIndex: "9999",
    display: "flex",
    flexDirection: "column",
    maxWidth: "300px",
    color: "white",
  });

  const textDiv = document.createElement("div");
  textDiv.textContent = text;
  Object.assign(textDiv.style, {
    fontSize: "14px",
    fontWeight: "500",
    color: "white",
    marginBottom: "8px",
  });
  popover.appendChild(textDiv);

  const askButton = document.createElement("button");
  askButton.textContent = "Find Meaning";
  Object.assign(askButton.style, {
    backgroundColor: "darkorange",
    color: "black",
    border: "none",
    borderRadius: "6px",
    padding: "6px 12px",
    cursor: "pointer",
    fontWeight: "bold",
    fontSize: "14px",
    alignSelf: "flex-end",
    transition: "opacity 0.3s ease"
  });

  if (explanation) {
    askButton.disabled = true;
    askButton.style.opacity = "0.6";
    askButton.style.cursor = "not-allowed";
  } else {
  askButton.addEventListener("click", async (e) => {
    e.stopPropagation();

    askButton.remove();
    isLoading = true;
    createOrUpdatePopover(text, x, y, "", isLoading);

    // chrome.runtime.sendMessage(
    //   {
    //     action: "StoreQuery",
    //     text: `"${text}" — explain its meaning in the context of documentation.`,
    //   },
    //   (response) => {
    //     console.log("Response from StoreQuery:", response);

    //     setTimeout(() => {
    //       currentExplanation = "This is a dummy explanation";
    //       isLoading = false;
    //       createOrUpdatePopover(text, x, y, currentExplanation, isLoading);
    //     }, 1200);
    //   }
    // );

    try {
        const response = await new Promise<{ explanation?: string }>((resolve) => {
          chrome.runtime.sendMessage(
            {
              action: "StoreQuery",
              text: `${text} — explain its meaning in the context of documentation.`,
            },
            (response) => {
              resolve(response);
            }
          );
        });
    
        currentExplanation = response?.explanation || "No explanation returned";
      } catch (error) {
        console.error("Error getting explanation:", error);
        currentExplanation = "Error fetching explanation.";
      }

      isLoading = false;
      createOrUpdatePopover(text, x, y, currentExplanation, isLoading);
  });
  }

  if (!isLoading) {
    popover.appendChild(askButton);
  } else {
    const loader = document.createElement("div");
    loader.textContent = "Loading...";
    Object.assign(loader.style, {
      fontSize: "13px",
      color: "orange",
      fontStyle: "italic",
    });
    popover.appendChild(loader);
  }

  if (explanation && !isLoading) {
    const responseDiv = document.createElement("div");
    responseDiv.textContent = explanation;
    Object.assign(responseDiv.style, {
      marginTop: "8px",
      backgroundColor: "#222",
      padding: "6px",
      borderRadius: "6px",
      fontSize: "14px",
      color: "white",
    });
    popover.appendChild(responseDiv);
  }

  document.body.appendChild(popover);
}
