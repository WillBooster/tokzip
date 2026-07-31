import { htmlModule } from '../generated/html.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(htmlModule);

export { htmlModule };
