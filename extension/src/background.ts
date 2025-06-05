
/// <reference types="chrome"/>

async function sendRequest(){
    try{
        const response =await fetch("http://127.0.0.1:5000",{
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: "Hello from Chrome Extension!" })
        })

        const data = await response.json();
        console.log("server:", data);
    }catch(e){
        console.error("Error in sendRequest:", e);
    }
}
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url && tab.url.includes("youtube.com/watch")) {
        console.log("Back scr", tab.url);

        const urlParams = new URLSearchParams(new URL(tab.url).search);
        const videoId = urlParams.get("v");

        if (videoId) {
            console.log("videoid", videoId);
        }
        
        chrome.tabs.sendMessage(tabId, { 
            action: "videoId", 
            videoId: videoId 
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("Error sending message:", chrome.runtime.lastError);
            } else {
                console.log("Response", response);
            }
        });

    } else if (changeInfo.status === "complete" && tab.url) {
        chrome.tabs.sendMessage(tabId, {
            action: "others",
            url: tab.url
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("Error sending message:", chrome.runtime.lastError);
            } else {
                console.log("Response", response);
            }
        });
    }
});
