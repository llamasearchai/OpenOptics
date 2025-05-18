class NetworkCommandCenter {
  constructor(apiClient) {
    this.apiClient = apiClient;
    this.commandHistory = [];
    this.initializeUI();
  }
  
  initializeUI() {
    const commandInput = document.getElementById('command-input');
    commandInput.addEventListener('keypress', async (e) => {
      if (e.key === 'Enter') {
        const command = commandInput.value;
        this.executeNaturalLanguageCommand(command);
        commandInput.value = '';
      }
    });
  }
  
  async executeNaturalLanguageCommand(command) {
    this.appendToCommandHistory(`> ${command}`);
    
    try {
      // Examples of natural language commands:
      // "Show me all spine switches with utilization above 80%"
      // "Optimize east zone for power efficiency and show before/after comparison"
      // "Alert me if any link between pod 3 and 4 experiences packet loss above 0.1%"
      
      const response = await this.apiClient.processCommand(command);
      
      if (response.type === 'visualization') {
        this.updateVisualization(response.data);
      } else if (response.type === 'table') {
        this.displayTable(response.data);
      } else if (response.type === 'text') {
        this.appendToCommandHistory(response.data);
      } else if (response.type === 'alert') {
        this.createAlert(response.data);
      }
    } catch (error) {
      this.appendToCommandHistory(`Error: ${error.message}`);
    }
  }
  
  appendToCommandHistory(text) {
    const historyElement = document.getElementById('command-history');
    const entry = document.createElement('div');
    entry.textContent = text;
    historyElement.appendChild(entry);
    historyElement.scrollTop = historyElement.scrollHeight;
    this.commandHistory.push(text);
  }
  
  // Additional methods for updating UI based on command results
}