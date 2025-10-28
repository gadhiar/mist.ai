let allConversations = [];

document.addEventListener('DOMContentLoaded', async () => {
  await loadConversations();

  document.getElementById('searchInput').addEventListener('input', () => {
    const query = document.getElementById('searchInput').value.toLowerCase();
    const filtered = allConversations.filter(c =>
      c.user_message.toLowerCase().includes(query) ||
      c.ai_message.toLowerCase().includes(query)
    );
    renderConversations(filtered);
  });

  document.getElementById('deleteAllBtn').addEventListener('click', async () => {
    if (!confirm("Are you sure you want to delete ALL conversations?")) return;
    await fetch('/api/conversations', { method: 'DELETE' });
    await loadConversations();
  });
});

async function loadConversations() {
  const res = await fetch('/api/conversations');
  allConversations = await res.json();
  renderConversations(allConversations);
}

function renderConversations(list) {
  const container = document.getElementById('conversationList');
  container.innerHTML = '';

  if (list.length === 0) {
    container.innerHTML = '<p class="text-gray-400">No conversations found.</p>';
    return;
  }

  list.forEach(conv => {
    const div = document.createElement('div');
    div.className = "bg-gray-800 p-4 rounded shadow";
    div.innerHTML = `
      <div><strong>User:</strong></div>
      <textarea data-id="${conv.id}" data-field="user" class="w-full p-2 rounded bg-gray-700 mt-1">${conv.user_message}</textarea>
      <div class="mt-2"><strong>AI:</strong></div>
      <textarea data-id="${conv.id}" data-field="ai" class="w-full p-2 rounded bg-gray-700 mt-1">${conv.ai_message}</textarea>
      <button class="saveBtn mt-2 bg-green-600 hover:bg-green-500 px-4 py-1 rounded">Save</button>
      <button class="deleteBtn mt-2 ml-2 bg-red-600 hover:bg-red-500 px-4 py-1 rounded">Delete</button>
    `;
    container.appendChild(div);

    div.querySelector('.saveBtn').addEventListener('click', async () => {
      const id = conv.id;
      const user = div.querySelector('textarea[data-field="user"]').value;
      const ai = div.querySelector('textarea[data-field="ai"]').value;
      await fetch(`/api/conversations/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_message: user, ai_message: ai })
      });
      alert("Saved.");
    });

    div.querySelector('.deleteBtn').addEventListener('click', async () => {
      const id = conv.id;
      if (!confirm("Delete this conversation?")) return;
      await fetch(`/api/conversations/${id}`, { method: 'DELETE' });
      await loadConversations();
    });
  });
}
